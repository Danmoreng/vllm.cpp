// FIX-KV-POOL-MIN-FIT (ISSUE-LOCAL-01M36QVG0KGKEP4MSMKT18MMZ4): the startup
// check that the KV pool can hold ONE request of `max_model_len` tokens must
// count every KV cache group, and the null block.
//
// Every case enters through the production constructor
// `LoadedEngine(config, weights, tokenizer, params, dflash_draft)`, the seam
// FromModelDir uses, so the check under test is the one a server runs
// (`LoadedEngine::ResolveMaxModelLen`), and the allocation under test is the
// scheduler's own (`KVCacheManager::allocate_slots` over one shared BlockPool).
//
// Mirrored from vllm @ e126687a9a:
//   vllm/v1/core/kv_cache_utils.py:2029-2058  _max_memory_usage_bytes_from_groups
//     -- a request claims blocks from EVERY group out of one pool, so the need
//        is the SUM of the per-group block counts;
//   vllm/v1/core/kv_cache_utils.py:2304-2327  the null block is reserved before
//     the check and the auto-fit;
//   vllm/v1/core/kv_cache_utils.py:830-867    the refusal and its message.
// Upstream's scheduler BREAKS out of the waiting loop when a request cannot
// allocate (vllm/v1/core/sched/scheduler.py:1091-1098) and never raises, so a
// request the startup check lets through waits forever. That is the defect:
// on dgx:gpu0 a 32k request sat for 23 minutes on an idle GPU.
//
// The geometry, at block_size 32 and a 128-token context:
//   fa        cdiv(128, 32)                       = 4
//   gdn       cdiv(128, 32) + k (k = 3)           = 7   (MambaManager, 'none')
//   fa_draft  cdiv(128, 32)                       = 4   (speculative config only)
//   null block                                    = 1
// so the speculative engine needs 16 blocks and the plain one needs 9. The
// pre-fix check asked for 4 in both cases.
#include <doctest/doctest.h>

#include <chrono>

#include "../v1/spec_decode/dflash2_runner_fixture.h"

#include "vllm/v1/core/kv_cache_manager.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/request.h"

namespace {

constexpr int kCtx = 128;         // max_model_len under test; 4 blocks of 32
constexpr int kSpecNeed = 4 + (4 + kSpecTokens) + 4 + 1;  // 16
constexpr int kPlainNeed = 4 + 4 + 1;                     // 9
constexpr int kDeadlineSeconds = 60;
// One token id of the tiny BPE fixture ("hello"); prompts go in pre-tokenized
// so the context length is an exact number.
constexpr int32_t kTok = 13;

// What one request did, with "never answered" kept apart from "finished",
// because the defect IS a request that never answers.
struct Outcome {
  bool finished = false;
  bool timed_out = false;
  std::string threw;
  size_t num_tokens = 0;
};

std::string Describe(const Outcome& o) {
  if (!o.threw.empty()) return "threw: " + o.threw;
  if (o.timed_out)
    return "NO OUTPUT within " + std::to_string(kDeadlineSeconds) +
           "s: admitted and never scheduled";
  if (o.finished) return "finished, " + std::to_string(o.num_tokens) + " tokens";
  return "no terminal state";
}

Outcome Drain(LoadedEngine& eng, int prompt_len, int max_tokens) {
  Outcome o;
  try {
    vllm::v1::AsyncLLM& async = eng.async_engine();
    vllm::v1::AsyncRequest req = async.add_request(
        "req", std::vector<int32_t>(static_cast<size_t>(prompt_len), kTok),
        Greedy(max_tokens));
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(kDeadlineSeconds);
    for (;;) {
      std::optional<vllm::RequestOutput> out =
          async.get_output_for(req, std::chrono::milliseconds(200));
      if (out.has_value()) {
        if (out->finished) {
          o.finished = true;
          if (!out->outputs.empty()) o.num_tokens = out->outputs[0].token_ids.size();
          return o;
        }
        continue;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        o.timed_out = true;
        try {
          async.abort("req");
        } catch (const std::exception&) {
        }
        return o;
      }
    }
  } catch (const std::exception& e) {
    o.threw = e.what();
  }
  return o;
}

EngineParams PlainParams(int max_model_len, int num_blocks) {
  EngineParams p;
  if (max_model_len > 0) p.max_model_len = max_model_len;
  p.max_num_seqs = 1;
  // One chunk holds the whole prompt, so the request's peak claim happens in
  // ONE allocate_slots call and is exactly the per-request maximum.
  p.max_num_batched_tokens = 8192;
  p.num_blocks = num_blocks;
  return p;
}

// Construct the speculative (three-group) engine. Returns the refusal text, or
// "" when construction succeeded, in which case `run` is called on the engine.
std::string BuildSpec(int max_model_len, int num_blocks,
                      const std::function<void(LoadedEngine&)>& run) {
  const HfConfig target = MakeDenseConfig(kCtx);
  const ScratchDraftDir dir;
  try {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir, max_model_len, /*max_num_seqs=*/1,
                                      /*max_num_batched_tokens=*/8192, num_blocks),
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
    run(eng);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

std::string BuildPlain(int max_model_len, int num_blocks,
                       const std::function<void(LoadedEngine&)>& run) {
  const HfConfig target = MakeDenseConfig(kCtx);
  try {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     PlainParams(max_model_len, num_blocks));
    run(eng);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

// The ROUTE predicate: can the scheduler's own allocator place one fresh
// request whose single allocation reaches `prompt_len + lookahead` tokens
// (capped at max_model_len)? The manager is built from the ENGINE's resolved
// KVCacheConfig with the arguments `Scheduler` passes
// (`src/vllm/v1/core/sched/scheduler.cpp`, `kv_cache_manager =`), and the
// lookahead is the engine's own `NumLookaheadTokens()`, which is what the
// scheduler hands `allocate_slots` for a waiting request.
//
// Why the allocator directly and not a request through the runner: a
// speculative request whose GDN claim exceeds the GDN block-table row
// (cdiv(max_model_len, 32), no `+ k`) writes past that row on the host
// (ISSUE-LOCAL-01M36YFXAHPXMCFAGWQABT6KCE). That is a separate defect this row
// records and does not fix, and a max-length speculative request is exactly
// the request that trips it.
bool AllocatorPlaces(const LoadedEngine& eng, int num_blocks, int prompt_len) {
  vllm::v1::KVCacheConfig cfg = eng.kv_cache_config();
  cfg.num_blocks = num_blocks;
  const int lookahead = eng.speculative_config().has_value()
                            ? eng.speculative_config()->NumLookaheadTokens()
                            : 0;
  vllm::v1::KVCacheManager mgr(
      cfg, eng.max_model_len(), /*scheduler_block_size=*/32,
      /*hash_block_size=*/32, /*max_num_batched_tokens=*/8192,
      eng.prefix_caching_enabled(), /*use_eagle=*/false, /*log_stats=*/false,
      /*enable_kv_cache_events=*/false, /*dcp_world_size=*/1,
      /*pcp_world_size=*/1, /*watermark=*/0.0);
  vllm::v1::Request req("route",
                        std::vector<int32_t>(static_cast<size_t>(prompt_len), kTok),
                        Greedy(1), /*arrival_time=*/0.0);
  return mgr.allocate_slots(req, prompt_len, /*num_new_computed_tokens=*/0,
                            std::nullopt, lookahead)
      .has_value();
}

}  // namespace

TEST_CASE(
    "kv pool min fit: a speculative pool one block short of three tables is "
    "REFUSED at startup") {
  // The benchmark's shape: fa + gdn + fa_draft. One block short of the sum.
  Outcome served;
  const std::string refusal = BuildSpec(kCtx, kSpecNeed - 1, [&](LoadedEngine& eng) {
    // Only reached when the check let the pool through. Show what the
    // admitted max-length request then does, so a red names the hang.
    served = Drain(eng, kCtx - 1, 1);
  });
  INFO("if admitted, the max-length request: ", Describe(served));
  REQUIRE_FALSE(refusal.empty());
  // Upstream's message (kv_cache_utils.py:857-866), with the pinned length.
  CHECK(refusal.find("To serve at least one request with the model's max seq "
                     "len (" + std::to_string(kCtx) + ")") != std::string::npos);
  CHECK(refusal.find("larger than the available KV cache memory") !=
        std::string::npos);
  // The estimate names a length the pool CAN serve: 96 tokens needs
  // 3 + (3 + 3) + 3 = 12 blocks <= 14 usable; 97 needs 15.
  CHECK(refusal.find("estimated maximum model length is 96") != std::string::npos);
}

TEST_CASE(
    "kv pool min fit: the refusal and the allocator agree at the speculative "
    "boundary") {
  // The smallest pool the check accepts is exactly the smallest pool on which
  // the scheduler's allocator can place a max-length request: 127 prompt tokens
  // plus the dflash lookahead (k + 1), capped at 128, claim 4 + 7 + 4 blocks.
  bool at_need = false;
  bool one_short = false;
  const std::string refusal = BuildSpec(kCtx, kSpecNeed, [&](LoadedEngine& eng) {
    REQUIRE(eng.max_model_len() == kCtx);
    REQUIRE(eng.kv_cache_config().kv_cache_groups.size() == 3);
    at_need = AllocatorPlaces(eng, kSpecNeed, kCtx - 1);
    one_short = AllocatorPlaces(eng, kSpecNeed - 1, kCtx - 1);
  });
  INFO("refusal: ", refusal);
  REQUIRE(refusal.empty());
  CHECK(at_need);
  // And one block fewer -- the pool test 1 shows the check refuses -- is one
  // the allocator cannot place the request on, i.e. the request would wait.
  CHECK_FALSE(one_short);
}

TEST_CASE(
    "kv pool min fit: a plain hybrid pool counts both tables and the null "
    "block") {
  // No speculative config: fa + gdn. One block short refuses...
  const std::string short_refusal = BuildPlain(kCtx, kPlainNeed - 1, [](LoadedEngine&) {});
  CHECK(short_refusal.find("max seq len (" + std::to_string(kCtx) + ")") !=
        std::string::npos);

  // ...and the counted need serves a max-length request.
  Outcome served;
  const std::string refusal = BuildPlain(kCtx, kPlainNeed, [&](LoadedEngine& eng) {
    served = Drain(eng, kCtx - 1, 1);
  });
  INFO("refusal: ", refusal);
  REQUIRE(refusal.empty());
  INFO("max-length request: ", Describe(served));
  CHECK(served.finished);
  CHECK_FALSE(served.timed_out);
}

TEST_CASE(
    "kv pool min fit: an unpinned max_model_len auto-fits to what three tables "
    "hold") {
  // _auto_fit_max_model_len (kv_cache_utils.py:2096-2157) over the same group
  // sum. 15 blocks = 14 usable: 96 tokens needs 3 + 6 + 3 = 12, 97 needs 15.
  int fitted = 0;
  bool places = false;
  const std::string refusal =
      BuildSpec(/*max_model_len=*/0, kSpecNeed - 1, [&](LoadedEngine& eng) {
        fitted = eng.max_model_len();
        places = AllocatorPlaces(eng, kSpecNeed - 1, fitted - 1);
      });
  INFO("refusal: ", refusal);
  REQUIRE(refusal.empty());
  CHECK(fitted == 96);
  // The fitted length is one the allocator can place.
  CHECK(places);
}

// FIX-BLOCK-TABLE-ROW-WIDTH (ISSUE-LOCAL-01M36YFXAHPXMCFAGWQABT6KCE): the
// worker's block-table row of each KV cache group is sized from that group's
// spec, through the production engine.
//
// THE DEFECT. The runner built every group's row at cdiv(max_model_len,
// block_size). The GDN group with a speculative config claims up to
// cdiv(max_model_len, block_size) + k blocks for one request
// (`MambaManager::get_num_blocks_to_allocate`), and `BlockTable::append_row`
// wrote the claim with no bounds check. On this fixture (max_model_len 128,
// block size 32, k = 3, one sequence) a 93-token prompt wrote three ids past
// the end of the one-row host buffer and glibc aborted the process on the heap
// corruption (`free(): invalid pointer` in the issue's run, `corrupted size vs.
// prev_size` in this row's red run).
//
// THE MIRROR (vLLM e126687a9a). `MambaSpec.max_num_blocks_per_req`
// (`vllm/v1/kv_cache_interface.py:896-905`) sizes a Mamba group's row with
// the `+ num_speculative_blocks`, the runner builds each group's width from
// its spec (`vllm/v1/worker/gpu_model_runner.py:7340-7360`), and a Mamba row is
// not rounded to 128 tokens (`vllm/v1/worker/block_table.py:322-331`).
//
// Every case enters through `LoadedEngine` and its `AsyncLLM`, the path the
// server takes. Its own binary because the DFlash2 fixture header keeps its
// helpers in an anonymous namespace per binary.
#include <doctest/doctest.h>

#include <chrono>

#include "../spec_decode/dflash2_runner_fixture.h"

#include "vllm/v1/core/kv_cache_manager.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/request.h"

namespace {

// The fixture's geometry. Block size 32 is `EngineParams::block_size`'s
// default, which is also the GDN group's block size in `none` mode here.
constexpr int kCtx = 128;
constexpr int kBlockSize = 32;
constexpr int kNumBlocks = 512;
// One token id of the tiny BPE fixture. Prompts go in pre-tokenized, so each
// length under test is exact.
constexpr int32_t kTok = 13;
constexpr int kDrainBudgetSeconds = 60;

struct Outcome {
  bool finished = false;
  bool timed_out = false;
  std::string threw;
  std::vector<int32_t> tokens;
};

std::string Describe(const Outcome& o) {
  if (!o.threw.empty()) return "threw: " + o.threw;
  if (o.timed_out) return "no terminal output within the deadline";
  if (o.finished) return "finished with " + std::to_string(o.tokens.size()) + " tokens";
  return "neither finished, timed out, nor threw";
}

// Drain one request with a deadline, so a dead engine reports instead of
// hanging the binary.
Outcome Drain(vllm::v1::AsyncLLM& async, int prompt_len, int max_tokens,
              const std::string& req_id) {
  Outcome o;
  try {
    vllm::v1::AsyncRequest req = async.add_request(
        req_id, std::vector<int32_t>(static_cast<size_t>(prompt_len), kTok),
        Greedy(max_tokens));
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(kDrainBudgetSeconds);
    for (;;) {
      std::optional<vllm::RequestOutput> out =
          async.get_output_for(req, std::chrono::milliseconds(200));
      if (out.has_value()) {
        if (out->finished) {
          o.finished = true;
          if (!out->outputs.empty()) o.tokens = out->outputs[0].token_ids;
          return o;
        }
        continue;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        o.timed_out = true;
        try {
          async.abort(req_id);
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

// Build the three-group speculative engine (fa, gdn, fa_draft) and run `body`.
void WithSpecEngine(const std::function<void(LoadedEngine&)>& body) {
  const HfConfig target = MakeDenseConfig(kCtx);
  const ScratchDraftDir dir;
  LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                   DflashSpecParams(dir, kCtx, /*max_num_seqs=*/1,
                                    /*max_num_batched_tokens=*/8192, kNumBlocks),
                   MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
  REQUIRE(eng.max_model_len() == kCtx);
  REQUIRE(eng.kv_cache_config().kv_cache_groups.size() == 3);
  body(eng);
}

}  // namespace

// ─── The issue's reproduction ───────────────────────────────────────────────
TEST_CASE("block table row width: a 93-token speculative prompt at max_model_len "
          "128 finishes") {
  // Pre-fix: the GDN claim is cdiv(min(93 + 4, 128) + 32 * 3, 32) = 7 blocks
  // against a 4-block row, and glibc aborts the process on the corrupted
  // heap (SIGABRT).
  Outcome first, second;
  WithSpecEngine([&](LoadedEngine& eng) {
    first = Drain(eng.async_engine(), 93, /*max_tokens=*/1, "req-93");
    // A second request on the same engine: the engine is still alive.
    second = Drain(eng.async_engine(), 8, /*max_tokens=*/2, "req-after");
  });
  INFO("93-token request: ", Describe(first));
  CHECK(first.finished);
  CHECK(first.threw.empty());
  CHECK(first.tokens.size() == 1);
  INFO("following request: ", Describe(second));
  CHECK(second.finished);
  CHECK(second.tokens.size() == 2);
}

// ─── The boundary ───────────────────────────────────────────────────────────
TEST_CASE("block table row width: the longest accepted prompt finishes and one "
          "token more is refused by name") {
  Outcome longest, one_more, after;
  WithSpecEngine([&](LoadedEngine& eng) {
    // max_model_len - 1 is the longest prompt the input processor accepts
    // (input_processor.cpp, input_processor.py:423-432). Its allocation reaches
    // max_model_len once the lookahead is added, which is the largest claim any
    // request can make: cdiv(128 + 32 * 3, 32) = 7 GDN blocks.
    longest = Drain(eng.async_engine(), kCtx - 1, /*max_tokens=*/1, "req-longest");
    one_more = Drain(eng.async_engine(), kCtx, /*max_tokens=*/1, "req-one-more");
    after = Drain(eng.async_engine(), 8, /*max_tokens=*/2, "req-after");
  });
  INFO("longest: ", Describe(longest));
  CHECK(longest.finished);
  CHECK(longest.threw.empty());
  CHECK(longest.tokens.size() == 1);

  INFO("one more: ", Describe(one_more));
  CHECK_FALSE(one_more.finished);
  CHECK(one_more.threw.find("plus the number of requested output tokens (at "
                            "least 1) is longer than the maximum model length "
                            "of 128") != std::string::npos);

  INFO("after: ", Describe(after));
  CHECK(after.finished);
  CHECK(after.tokens.size() == 2);
}

// ─── The widths the runner builds, and the claim they must hold ─────────────
TEST_CASE("block table row width: the engine's GDN row is exactly the largest "
          "GDN claim") {
  using vllm::v1::KVCacheSpecKind;
  using vllm::v1::SlotMappingMode;
  WithSpecEngine([&](LoadedEngine& eng) {
    const vllm::v1::KVCacheConfig& cfg = eng.kv_cache_config();
    // The geometry the runner passes to its InputBatch.
    const vllm::v1::BlockTableGeometry geo =
        vllm::v1::block_table_geometry(cfg, eng.max_model_len());
    REQUIRE(geo.max_num_blocks.size() == 3);
    int gdn = -1;
    for (size_t g = 0; g < cfg.kv_cache_groups.size(); ++g) {
      const bool mamba =
          cfg.kv_cache_groups[g].kv_cache_spec->kind() == KVCacheSpecKind::kMamba;
      INFO("group ", g);
      CHECK(geo.block_sizes[g] == kBlockSize);
      if (mamba) {
        gdn = static_cast<int>(g);
        // cdiv(128, 32) + k, and not rounded.
        CHECK(geo.max_num_blocks[g] == 4 + kSpecTokens);
        CHECK(geo.slot_mapping_modes[g] == SlotMappingMode::kNone);
      } else {
        // fa and fa_draft keep cdiv(128, 32) = 4.
        CHECK(geo.max_num_blocks[g] == 4);
        CHECK(geo.slot_mapping_modes[g] == SlotMappingMode::kTokenToKvSlot);
      }
    }
    REQUIRE(gdn >= 0);

    // The scheduler's own allocator (built as scheduler.cpp builds it) placing
    // the longest accepted prompt with the engine's lookahead claims exactly
    // the GDN row width: the boundary has zero slack.
    const int lookahead = eng.speculative_config()->NumLookaheadTokens();
    vllm::v1::KVCacheManager mgr(
        cfg, eng.max_model_len(), /*scheduler_block_size=*/kBlockSize,
        /*hash_block_size=*/kBlockSize, /*max_num_batched_tokens=*/8192,
        eng.prefix_caching_enabled(), /*use_eagle=*/false, /*log_stats=*/false,
        /*enable_kv_cache_events=*/false, /*dcp_world_size=*/1,
        /*pcp_world_size=*/1, /*watermark=*/0.0);
    vllm::v1::Request req(
        "claim", std::vector<int32_t>(static_cast<size_t>(kCtx - 1), kTok),
        Greedy(1), /*arrival_time=*/0.0);
    const auto blocks =
        mgr.allocate_slots(req, kCtx - 1, 0, std::nullopt, lookahead);
    REQUIRE(blocks.has_value());
    const std::vector<std::vector<int>> ids = mgr.get_block_ids("claim");
    const int row_width = vllm::v1::get_block_table_width(
        geo.max_num_blocks[static_cast<size_t>(gdn)], kBlockSize, std::nullopt,
        std::nullopt);
    CHECK(static_cast<int>(ids[static_cast<size_t>(gdn)].size()) == row_width);
    CHECK(row_width == 7);
  });
}

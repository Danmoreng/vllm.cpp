# Count every KV cache group when the pool is checked at startup

Row: `FIX-KV-POOL-MIN-FIT`. Issue: `ISSUE-LOCAL-01M36QVG0KGKEP4MSMKT18MMZ4`.
Parity pin: vLLM `e126687a9a` (`.agents/upstream-sync.md`).

## Scope

The startup check that a pool can hold one request of `max_model_len` tokens
counts one block table. The scheduler allocates one block table for each KV
cache group, and every group draws its block ids from the same pool. A Qwen3.5
or Qwen3.6 configuration with a speculative config has three groups: `fa`,
`gdn` and `fa_draft`. A configuration without one has two. The check therefore
accepts a pool that holds about one third of what a max-length request needs.
The request is then admitted, the waiting loop cannot allocate it, the loop
breaks without an error, and the request waits forever with an idle device.

The check also counts every block in the pool as usable. `BlockPool` keeps
block 0 as the null block, so only `num_blocks - 1` blocks are usable. A pool of
exactly the needed size passes the check and cannot serve the request.

This row makes the check, and the auto-fit of an unpinned `max_model_len`, count
blocks the way the allocator claims them:

- the sum of the per-group block counts of one max-length request, and
- the pool minus the null block.

Out of scope:

- The size of the Mamba/GDN group's claim itself. See `## Design`, the
  "Recorded deviation" part, and `## Owed`.
- The recurrent-state budget (`hybrid_kv_budget.h`) and the state-memory check.
  Both bound a different allocation and do not change.
- A scheduler-side error for a request that cannot fit. vLLM has none. See
  `## Design`.

## Anchors

Upstream, all at `e126687a9a`:

| What | Where |
|---|---|
| The refusal and its message | `vllm/v1/core/kv_cache_utils.py:830-867` (`_check_enough_kv_cache_memory`) |
| Needed bytes: sum of per-group blocks times the pool's bytes per block | `vllm/v1/core/kv_cache_utils.py:2029-2058` (`_max_memory_usage_bytes_from_groups`) |
| Estimated length for the message and auto-fit | `vllm/v1/core/kv_cache_utils.py:2061-2093` (`_estimate_max_model_len_from_groups`) |
| Auto-fit of an unpinned length | `vllm/v1/core/kv_cache_utils.py:2096-2157` (`_auto_fit_max_model_len`) |
| Null block reserved before the check and auto-fit | `vllm/v1/core/kv_cache_utils.py:2304-2327`, and `:953-961` in the public `check_enough_kv_cache_memory` |
| Full attention per-request usage | `vllm/v1/kv_cache_interface.py:464-469` |
| Chunked-local and sliding-window usage (shared with the admission cap) | `vllm/v1/kv_cache_interface.py:658-681`, `:704-738` |
| Mamba per-request usage, by `mamba_cache_mode` | `vllm/v1/kv_cache_interface.py:883-894` |
| Mamba block size in `none` mode is `max_model_len` | `vllm/model_executor/models/config.py:645-657` |
| Mamba allocation in `none` mode (`cdiv(num_tokens + bs*k, bs)`) | `vllm/v1/core/single_type_kv_cache_manager.py:1579-1612` |
| The waiting loop breaks when a request cannot allocate | `vllm/v1/core/sched/scheduler.py:1091-1098` |
| Upstream tests ported | `tests/v1/core/test_kv_cache_utils.py:2754-2776` (hybrid auto-fit), `:3164-3186`, `:3189-3207`, `:3210-3229` (null block) |

Local:

| What | Where |
|---|---|
| Pinned check and auto-fit | `src/vllm/entrypoints/model_loader.cpp::ResolveMaxModelLen` |
| Group construction with a speculative config | `src/vllm/model_executor/models/qwen3_5_common.cpp::MakeQwen3_5KVCacheSpec` |
| Pool bytes per block id | `src/vllm/v1/kv_cache_interface.cpp::KVBytesPerBlock` |
| One manager per group, one shared pool | `src/vllm/v1/core/kv_cache_coordinator.cpp:130-165` |
| Null block | `src/vllm/v1/core/block_pool.cpp:53-57` |
| Mamba allocation | `src/vllm/v1/core/single_type_kv_cache_manager.cpp::MambaManager::get_num_blocks_to_allocate` |
| Sliding-window and chunked-local admission caps | `src/vllm/v1/kv_cache_interface.cpp:195-207` |
| Waiting loop break | `src/vllm/v1/core/sched/scheduler.cpp:927-940` |

## Design

### What vLLM does

vLLM refuses at startup. `get_kv_cache_configs` subtracts one block's bytes for
the null block, then calls `_check_enough_kv_cache_memory` with
`_max_memory_usage_bytes_from_groups`. That function multiplies the pool's
bytes per block by the SUM over groups of
`cdiv(spec.max_memory_usage_bytes, spec.page_size_bytes)`. The comment at
`kv_cache_utils.py:2040-2041` states the reason: each group claims its own
blocks from the shared pool.

If a request that cannot fit is admitted anyway, the upstream scheduler breaks
out of the waiting loop and the request waits (`scheduler.py:1091-1098`). vLLM
raises no error there. The startup check is the whole guard, so this row mirrors
the startup check and does not add a scheduler error.

### The port

A new function `max_blocks_per_request(spec, max_model_len,
max_num_batched_tokens)` returns upstream's
`cdiv(max_memory_usage_bytes, page_size_bytes)` for one group:

| Spec kind | Blocks for one request |
|---|---|
| Full attention, MLA | `cdiv(max_model_len, block_size)` |
| Sliding window, sliding-window MLA, chunked local | the spec's existing `max_admission_blocks_per_request`, the function the manager's admission cap already calls |
| Mamba, `align` | `2 + num_speculative_blocks` |
| Mamba, `none` or `all` | `cdiv(max_model_len, block_size) + num_speculative_blocks` |

`max_memory_usage_bytes_from_groups(cfg, len, mnbt)` returns
`KVBytesPerBlock(cfg) * sum(max_blocks_per_request(group))`, which is
`_max_memory_usage_bytes_from_groups`. `estimate_max_model_len_from_groups`
is the upstream binary search over that function.
`ResolveMaxModelLen` subtracts one block's bytes from the pool for the null
block and uses the two functions for both the pinned check and the auto-fit. The
message is the existing `check_enough_kv_cache_memory` text, which is
upstream's wording plus the local flags.

`max_num_batched_tokens` enters only the sliding-window and chunked-local
formulas. `ResolveMaxModelLen` gets a fifth parameter, the dense-architecture
flag, so that it can compute the same `ResolveMaxNumBatchedTokens` value the
scheduler receives. For auto-fit it computes that value at the derived length.
The budget can only shrink at a shorter length, so this is the conservative
direction.

### Recorded deviation: the Mamba `none` formula

Upstream's `none` formula is `(1 + num_speculative_blocks) * page_size_bytes`.
That value is correct upstream only because upstream sets the Mamba block size
to `max_model_len` in `none` mode (`config.py:657`). The allocator then claims
`cdiv(num_tokens + bs * k, bs) = 1 + k` blocks.

This tree builds the `none`-mode GDN spec at the attention block size, 32
tokens. The same allocator arithmetic (`MambaManager::get_num_blocks_to_allocate`)
then claims `cdiv(num_tokens, 32) + k` blocks. A literal port of `1 + k` would
leave the defect in place. At `max_model_len 32768` it counts
`2 * 1024 + 1 + k` blocks against a real claim of `3 * 1024 + k`, so the
2,114-block pool that hung would still pass for any `k` up to 64.

The check must count what the allocator claims, so the port uses
`cdiv(max_model_len, block_size) + k`. At upstream's block size this expression
is `1 + k`, which is upstream's value. At ours it is our allocator's claim. The
ported upstream hybrid test therefore builds its Mamba spec at upstream's
resolved `none`-mode geometry, `block_size = max_model_len`, and keeps upstream's
numbers.

Moving the GDN group to upstream's block size would shrink the claim to `1 + k`.
It changes the allocation of every hybrid registry and the runner's GDN block
handling, so it is a shared-seam change. `## Owed` records it.

### Existing tests that encoded the defect

`test_loaded_engine_dense.cpp` builds engines with `--num-blocks 1` and a
32-token context. That pool has zero usable blocks, so no request can run.
Those cases now use the smallest pool that serves the context. The fixture is a
hybrid model, so that pool is three blocks: one per group and the null block.

## Risks

- A configuration that started before now refuses. That is the intended
  change. Each refusal names the needed and available sizes, the estimated
  length, and the flags to change.
- Sliding-window models with a pinned length can refuse where they passed,
  because upstream adds one block per sliding group. The value comes from the
  same function as the manager's admission cap, so a refused configuration is
  one the cap would also block.
- The Mamba `none` formula over-counts compared to upstream geometry. It is
  exact for this tree's allocator. `## Owed` records the cost.

## Tests

Red first, through the production constructor
(`LoadedEngine(config, weights, tokenizer, params, dflash_draft)`, the seam
`FromModelDir` uses), in a new target `test_kv_pool_min_fit`:

1. A DFlash2 speculative engine (three groups: `fa`, `gdn`, `fa_draft`) with a
   pinned `max_model_len` and a pool one block short of the three-group need
   refuses at construction with `max seq len`. On the pre-fix tree it
   constructs. This is the red.
2. The same engine with one more block constructs and serves a request whose
   allocation reaches `max_model_len` in one step, to completion, within a
   deadline. The refusal and the allocator agree at the boundary.
3. The same boundary for a two-group model (no speculative config).
4. An unpinned `max_model_len` auto-fits to a length the three-group pool can
   serve, and that length is served.

Unit tests ported from upstream into `tests/vllm/v1/test_kv_cache_utils.cpp`:

- `test_auto_fit_max_model_len_with_hybrid`, adapted as `## Design` states.
- `test_check_enough_kv_cache_memory_reserves_null_block`.
- `test_auto_fit_max_model_len_reserves_null_block`.
- Per-kind `max_blocks_per_request` values, including the agreement of the
  Mamba formula with `MambaManager::get_num_blocks_to_allocate` for a fresh
  request.

Reachability: delete the production call in `ResolveMaxModelLen` in a scratch
copy, rebuild, and show that test 1 reds. Restore byte for byte, with a sha256
before and after.

## Gates

- `test_kv_pool_min_fit`, `test_loaded_engine_dense`, `test_kv_cache_utils`,
  `test_kv_state_budget`, `test_dflash2_ctx_capacity`, and the other test
  targets that construct a `LoadedEngine`, built with `-j 4` on the CPU tier.
- `scripts/agent-preflight.sh --staged`, judged by its verdict lines.

## Stop conditions

- vLLM at the pin does not refuse at startup: mirror vLLM and report it. (It
  does refuse.)
- The fix needs the GDN block-size change: return `NEEDS_DECISION`. (It does
  not; the check counts the current claim.)

## Owed

- `ISSUE-LOCAL-01M36XJNF0TRNZBH7GYCW756AQ`: the GDN group in `none` mode is
  built at the attention block size instead of `max_model_len`
  (`config.py:657`), so each request claims `cdiv(len, 32) + k` pool blocks
  where upstream claims `1 + k`. At `max_model_len 32768` with a speculative
  config, one request needs `3 * 1024 + k` blocks here and `2 * 1024 + 1 + k`
  at upstream geometry. This row counts the claim correctly and does not
  change it. Moving the block size is a shared-seam change across every hybrid
  registry and needs a developer decision.
- A pure recurrent model has `KVBytesPerBlock == 0`, and the check is skipped for
  it as before. Its Mamba group still claims pool blocks.

## Now

`ACTIVE` on `row/FIX-KV-POOL-MIN-FIT`.

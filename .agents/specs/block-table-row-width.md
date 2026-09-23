# Size each block-table row from its KV cache spec

Row: `FIX-BLOCK-TABLE-ROW-WIDTH`. Issue: `ISSUE-LOCAL-01M36YFXAHPXMCFAGWQABT6KCE`.
Parity pin: vLLM `e126687a9a` (`.agents/upstream-sync.md`).

## Scope

The worker's `MultiGroupBlockTable` sizes the row of every KV cache group at
`cdiv(max_model_len, block_size)`, rounded up to a 128-token multiple. The
runner passes no per-group width (`runner.cpp`, the two `input_batch_(...)`
initializers of `GPUModelRunner`). A Mamba/GDN group with a speculative config
claims up to `cdiv(max_model_len, block_size) + k` blocks for one request, so the
claim can be longer than the row. `BlockTable::append_row` does not check the
row capacity. The write then goes into the next request's row, or past the end
of the host buffer for the last row.

This row makes two changes:

1. The runner sizes each group's row from the group's spec, as vLLM does:
   `KVCacheSpec::max_num_blocks_per_req(max_len)`, and a Mamba group's width is
   not rounded to 128 tokens.
2. `BlockTable::append_row` refuses a write past the row capacity with a named
   `std::runtime_error`, before it changes any state.

Models and backends touched: every model that publishes a `MambaSpec` group.
These are the Qwen3.5/3.6 hybrids (`qwen3_5_common.cpp`), Kimi Linear, GLM-5
Next, Qwen4-Exp and Nemotron-H. The change is in the host-side block table of
the shared worker seam, so it applies to all backends (CPU, CUDA, ROCm, Vulkan,
Metal) in the same way. Attention groups (full, MLA, sliding window, chunked
local, the DFlash2 draft group) keep their current width for every block size
that divides 128 or is a multiple of it. See "Attention rounding" for the other
block sizes.

Out of scope:

- The GDN group's block size in `none` mode
  (`ISSUE-LOCAL-01M36XJNF0TRNZBH7GYCW756AQ`). This row sizes the row for the
  block size the group has now. See "Recorded deviation".
- `SlotMappingMode::kNone` skipping `compute_slot_mapping`. See "Recorded
  deviation".
- The startup pool check. `FIX-KV-POOL-MIN-FIT` owns it.

## Upstream anchors (vLLM `e126687a9a`)

- `vllm/v1/kv_cache_interface.py:197-207`: `KVCacheSpec.max_num_blocks_per_req`
  returns `cdiv(max_len, block_size)`.
- `vllm/v1/kv_cache_interface.py:432-435`: `AttentionSpec` divides by
  `block_size * decode_context_parallel_size`. This tree has no DCP, so the
  value is the base value.
- `vllm/v1/kv_cache_interface.py:883-905`: `MambaSpec.max_memory_usage_bytes`
  and `MambaSpec.max_num_blocks_per_req`. `align` returns
  `cdiv(max_len, block_size) + num_speculative_blocks`. `all` and `none` return
  `cdiv(max_memory_usage_bytes, page_size_bytes)`, which is
  `cdiv(max_model_len, block_size) + k` for `all` and `1 + k` for `none`.
- `vllm/v1/worker/gpu_model_runner.py:7340-7360`: the runner builds
  `max_num_blocks` from `spec.max_num_blocks_per_req(vllm_config, max_model_len)`
  and marks a Mamba group `SlotMappingMode.NONE`.
- `vllm/v1/worker/gpu/model_runner.py:555-577`: the same per-group width in the
  model runner V2, with `token_alignment=None` for a Mamba group.
- `vllm/v1/worker/block_table.py:29-49`: `get_block_table_width`.
- `vllm/v1/worker/block_table.py:322-331`: `MultiGroupBlockTable` rounds a
  `TOKEN_TO_KV_SLOT` group to 128 tokens and leaves a `NONE` group unrounded.
- `vllm/v1/worker/gpu/block_table.py:113-133`: `BlockTables.append_block_ids`
  raises `RuntimeError("Block table write for request {r}, group {i} exceeds
  row capacity ({end} > {row_capacity})")` before it writes. The V1 table
  (`block_table.py:157-173`) assigns a NumPy slice, which also raises when the
  ids do not fit.

## Design

`KVCacheSpec` gets a virtual `max_num_blocks_per_req(int max_len)`. The base
returns `cdiv(max_len, block_size)`. `MambaSpec` overrides it.

`vllm::v1::SlotMappingMode` (`kTokenToKvSlot`, `kNone`) and
`get_block_table_width` are ported to `block_table.h`. `MultiGroupBlockTable`
takes an optional `slot_mapping_modes` list and uses it to select the width
rounding, as `block_table.py:322-331` does. When the list is absent every group
is `kTokenToKvSlot`, which is the current rounding.

A new function `block_table_geometry(const KVCacheConfig&, int max_model_len)`
returns the per-group `max_num_blocks` and `slot_mapping_modes`. It is the loop
at `gpu_model_runner.py:7340-7360`. Both `GPUModelRunner` constructors pass its
result to `InputBatch`, which forwards it to `MultiGroupBlockTable`.

`BlockTable::append_row` computes `end = start + len(ids)` and throws before any
write when `end > max_num_blocks_per_req`. The message is upstream's, with the
group index that `MultiGroupBlockTable` records on each table.

### Attention rounding

The previous port rounded a row to a multiple of `128 / block_size` blocks,
and only when `block_size <= 128`. The pinned `get_block_table_width` rounds to
a multiple of `128 / gcd(128, block_size)` blocks for every block size. The two
agree when the block size divides 128 or is a multiple of it. For other sizes
the pinned rule gives a wider row: block size 48 at `max_model_len 256` goes
from 6 to 8 blocks, and block size 528 now rounds to 8 blocks. A wider row
cannot overflow, so this change only adds unused capacity. The existing case in
`test_block_table.cpp` that encoded the old value for block size 48 now asserts
the pinned value.

### Recorded deviation: the Mamba `none` width

Upstream's `none` width is `1 + k`. That value is correct upstream because
upstream builds the `none`-mode Mamba group at `block_size = max_model_len`
(`config.py:657`). `MambaManager` then claims `cdiv(num_tokens + bs * k, bs)`,
which is at most `1 + k`.

This tree builds the `none`-mode GDN group at the attention block size, 32
tokens (`ISSUE-LOCAL-01M36XJNF0TRNZBH7GYCW756AQ`). The same manager arithmetic
(`single_type_kv_cache_manager.cpp`, `MambaManager::get_num_blocks_to_allocate`)
claims up to `cdiv(max_model_len, 32) + k`. A literal `1 + k` row would refuse
every request longer than one block.

The port therefore returns `cdiv(max_len, block_size) + k` for `none`. When
`block_size >= max_len`, which is upstream's geometry, this is `1 + k`, which is
upstream's value. When the GDN block size moves to upstream's value, this
expression gives upstream's width with no further change. `FIX-KV-POOL-MIN-FIT`
uses the same expression for the pool check, for the same reason.

The row sizing is therefore expressible without the block-size change, and this
row does not return `NEEDS_DECISION`.

### Recorded deviation: `kNone` still computes a slot mapping

Upstream's `BlockTable.compute_slot_mapping` returns early for a `NONE` group
(`block_table.py:208-211`). This row ports `SlotMappingMode` for the width
selection only. `prepare_inputs.cpp` copies each group's slot mapping into the
step inputs, so skipping the GDN group's computation changes what the runner
receives. That is a separate change. The existing computation stays in range
with the new width, because a position below `max_model_len` indexes a block
below `cdiv(max_model_len, block_size)`.

## Risks

- A Mamba group's row gets narrower without speculation, because the 128-token
  rounding goes. The claim without speculation is at most
  `cdiv(max_model_len, block_size)`, which is the new width, and the bounds
  check makes any miscount a named error.
- Any consumer that assumes a GDN row width of `cdiv(max_model_len, bs)`
  rounded to 128 tokens. The runner reads the width from the table
  (`gather_block_table`), and `remap_gdn_state_slots` takes it as a parameter.
- The bounds check changes silent corruption into an exception inside the
  engine step. That is the intent.

## Tests

Red first, in a new target `test_block_table_row_width`, through the production
engine (`LoadedEngine(config, weights, tokenizer, params, dflash_draft)` and its
`AsyncLLM`, the path the server takes), with the three-group DFlash2 fixture
(`fa`, `gdn`, `fa_draft`; `k = 3`), `max_model_len 128`, block size 32,
512 blocks and one sequence:

1. A 93-token prompt with `max_tokens 1` finishes. On the current tree glibc
   aborts the process on the corrupted heap (`free(): invalid pointer` in the
   issue's run). This is the red.
2. The boundary. The longest prompt that the engine accepts finishes, and the
   GDN claim of that request equals the GDN row width exactly. One token more
   gets the engine's named length refusal and no crash.
3. `block_table_geometry` on the engine's resolved `KVCacheConfig`:
   `fa` and `fa_draft` keep `cdiv(128, 32) = 4` and `kTokenToKvSlot`, and `gdn`
   gets `4 + 3 = 7` and `kNone`.

Unit tests in `tests/vllm/v1/worker/test_block_table.cpp`:

4. `append_row` to exactly the row capacity succeeds. One block more throws the
   named error and leaves the row count and contents unchanged. The same for
   `add_row` and for a hybrid (`kernel_block_size < block_size`) table.
5. `MultiGroupBlockTable` rounds a `kTokenToKvSlot` group to 128 tokens and
   leaves a `kNone` group at its given width. `get_block_table_width` values
   from `block_table.py:29-49`.
6. In `tests/vllm/v1/test_kv_cache_interface.cpp`,
   `MambaSpec::max_num_blocks_per_req` for `none`, `align` and `all`, with and
   without speculative blocks, at this tree's geometry and at upstream's
   geometry (`block_size = max_model_len`, which gives `1 + k`).

Reachability: in a scratch copy, delete the `block_table_geometry` arguments
from the runner's `InputBatch` construction, rebuild, and show that tests 1 and
2 go red. Restore byte for byte, with a sha256 before and after.

## Gates

- `test_block_table_row_width` and every test target that constructs the
  runner, `InputBatch` or `MultiGroupBlockTable`, found with `ninja -t targets`
  and a grep for the constructors. CPU build with `-j 3`.
- `scripts/agent-preflight.sh --staged`, judged by its verdict lines.

## Stop conditions

- The correct row width needs the GDN block-size change: return
  `NEEDS_DECISION`. (It does not. See "Recorded deviation".)
- vLLM at the pin writes past the row silently: mirror vLLM and report it. (It
  does not. Both tables raise.)

## Owed

Nothing owed by this row. The GDN block size that this row's `none` width
accommodates is owned under `## Owed` in `.agents/specs/kv-pool-min-fit.md`.

## Now

`ACTIVE` on `row/FIX-BLOCK-TABLE-ROW-WIDTH`. The implementation and its CPU
gates are on the branch and wait for a fresh review. Two existing
`test_runner.cpp` cases encoded the old widths: one fed two block ids to a
group whose row holds one (an overflow the bounds check now refuses), and one
asserted the 128-token rounding on the recurrent group. Both now assert the
pinned behaviour.

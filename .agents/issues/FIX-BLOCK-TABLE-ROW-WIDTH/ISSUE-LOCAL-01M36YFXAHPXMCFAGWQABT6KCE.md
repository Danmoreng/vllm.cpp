ID: ISSUE-LOCAL-01M36YFXAHPXMCFAGWQABT6KCE
Title: engine: a speculative request near max_model_len writes past its GDN block-table row
Row: FIX-BLOCK-TABLE-ROW-WIDTH
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

MultiGroupBlockTable sizes every group's row at cdiv(max_model_len, block_size) rounded up to a multiple of 128/block_size (src/vllm/v1/worker/gpu/block_table.cpp:196-217, called with no max_num_blocks from the runner's InputBatch at src/vllm/v1/worker/gpu/runner.cpp:447 and :557). The none-mode GDN group is built at the attention block size, and MambaManager claims cdiv(num_tokens + bs*k, bs) blocks for it, up to cdiv(max_model_len, bs) + k. BlockTable::append_row does not bound-check, so a request whose claim exceeds the row width writes past the row: into the next request's row, or past the end of the host buffer for the last row. vLLM sizes each group's row from its spec, MambaSpec.max_num_blocks_per_req = cdiv(max_memory_usage_bytes, page_size_bytes) (vllm/v1/kv_cache_interface.py:896-905 @ e126687a9a), which includes the + k. Reproduction on the CPU tier, found by FIX-KV-POOL-MIN-FIT: the DFlash2 fixture (tests/vllm/v1/spec_decode/dflash2_runner_fixture.h, k = 3, lookahead 4) at max_model_len 128, block_size 32, 512 blocks, one request of 93 or more prompt tokens and max_tokens 1 aborts with 'free(): invalid pointer' in InputBatch::~InputBatch; 92 tokens and below completes. At max_model_len 32768 with a speculative config the same overflow happens for a prompt within about k blocks of max_model_len. The fix is a runner (shared worker seam) change and needs a developer decision.

## Resolution

2026-09-23, row FIX-BLOCK-TABLE-ROW-WIDTH (.agents/specs/block-table-row-width.md). The runner now builds each KV cache group's block-table width from the group's spec (block_table_geometry, gpu_model_runner.py:7340-7360 @ e126687a9a): KVCacheSpec::max_num_blocks_per_req is cdiv(max_len, block_size), MambaSpec's is cdiv(max_len, block_size) + num_speculative_blocks (kv_cache_interface.py:896-905, with the recorded none-mode deviation for this tree's GDN block size), and a kNone (Mamba) row is not rounded to 128 tokens (block_table.py:322-331). BlockTable::append_row refuses a write past the row with upstream's 'Block table write for request R, group G exceeds row capacity (end > cap)' before any state changes (gpu/block_table.py:125-131). Red on the unfixed tree: test_block_table_row_width, DFlash2 fixture at max_model_len 128, a 93-token prompt and the 127-token prompt both SIGABRT (glibc 'corrupted size vs. prev_size'). Green: 3/3 cases, 32/32 assertions; the 127-token prompt finishes, its GDN claim equals the 7-block row exactly, and a 128-token prompt gets the input processor's named length refusal. Reachability: with the runner's geometry arguments deleted the two engine cases go red with the named row-capacity error (7 > 4) instead of an abort.

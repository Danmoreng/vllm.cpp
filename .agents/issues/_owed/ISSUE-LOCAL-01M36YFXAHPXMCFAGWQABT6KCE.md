ID: ISSUE-LOCAL-01M36YFXAHPXMCFAGWQABT6KCE
Title: engine: a speculative request near max_model_len writes past its GDN block-table row
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

MultiGroupBlockTable sizes every group's row at cdiv(max_model_len, block_size) rounded up to a multiple of 128/block_size (src/vllm/v1/worker/gpu/block_table.cpp:196-217, called with no max_num_blocks from the runner's InputBatch at src/vllm/v1/worker/gpu/runner.cpp:447 and :557). The none-mode GDN group is built at the attention block size, and MambaManager claims cdiv(num_tokens + bs*k, bs) blocks for it, up to cdiv(max_model_len, bs) + k. BlockTable::append_row does not bound-check, so a request whose claim exceeds the row width writes past the row: into the next request's row, or past the end of the host buffer for the last row. vLLM sizes each group's row from its spec, MambaSpec.max_num_blocks_per_req = cdiv(max_memory_usage_bytes, page_size_bytes) (vllm/v1/kv_cache_interface.py:896-905 @ e126687a9a), which includes the + k. Reproduction on the CPU tier, found by FIX-KV-POOL-MIN-FIT: the DFlash2 fixture (tests/vllm/v1/spec_decode/dflash2_runner_fixture.h, k = 3, lookahead 4) at max_model_len 128, block_size 32, 512 blocks, one request of 93 or more prompt tokens and max_tokens 1 aborts with 'free(): invalid pointer' in InputBatch::~InputBatch; 92 tokens and below completes. At max_model_len 32768 with a speculative config the same overflow happens for a prompt within about k blocks of max_model_len. The fix is a runner (shared worker seam) change and needs a developer decision.

## Resolution

-

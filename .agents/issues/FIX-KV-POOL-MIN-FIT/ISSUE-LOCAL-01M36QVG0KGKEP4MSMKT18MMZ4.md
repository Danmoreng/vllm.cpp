ID: ISSUE-LOCAL-01M36QVG0KGKEP4MSMKT18MMZ4
Title: engine: a request the hybrid KV pool can never hold waits forever instead of being refused
Row: FIX-KV-POOL-MIN-FIT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

On Qwen3.8-27B EXL3 with a DFlash2 draft (tree d4738d241, dgx:gpu0, leases 55f0238f and 68ae4d65), a server started with --max-model-len 16384 --num-blocks 1024 --max-num-seqs 1 accepted a 15,624-token chat request and never returned it. The client timed out at 300 s, the server stayed alive, and nvidia-smi GPU utilization read 0 for every sample, while a working request on the same box read nonzero. The same prompt completed in 26.9 s on a 17,408-block pool. An 8,305-token prompt completed on the 1,024-block pool. The startup check admitted the configuration: check_enough_kv_cache_memory sizes the pool from KVBytesPerBlock, which counts only the attention groups (kv_cache_interface.cpp:241-266). The scheduler's MambaManager in the default 'none' mode requests cdiv(num_tokens, block_size) blocks for the GDN group as well (single_type_kv_cache_manager.cpp:788-795), so one request needs about twice the blocks the check assumed. When allocate_slots cannot place a request, the waiting loop breaks without an error (scheduler.cpp:929-940), so the request waits forever. Expected: either refuse the configuration at startup, or reject the request with an error. The upstream vLLM comparison of the none-mode Mamba allocation is not yet made.

## Resolution

2026-09-23, row/FIX-KV-POOL-MIN-FIT. LoadedEngine::ResolveMaxModelLen now refuses at startup, with vLLM's message, a pool that cannot hold one max_model_len request counted over EVERY KV cache group plus the null block (vllm/v1/core/kv_cache_utils.py:2029-2058, :2304-2327, :830-867 @ e126687a9a); the unpinned auto-fit counts the same way. vLLM's scheduler raises nothing for an unplaceable request (scheduler.py:1091-1098), so the startup check is the whole guard, as upstream. Red on the pre-fix tree (test_kv_pool_min_fit): the DFlash2 three-group engine at 15 blocks for max_model_len 128 constructed and its 127-token request gave NO OUTPUT within 60s; auto-fit chose 128 where 96 fits. Green after: 4/4 cases, 16 assertions; test_kv_cache_utils 40/40; the 107 affected CPU test targets pass (test_qwen35_paged_engine skipped for weights). Deleting the production check call reds the case. Two related defects are filed, not fixed: ISSUE-LOCAL-01M36XJNF0TRNZBH7GYCW756AQ (GDN none-mode block size) and ISSUE-LOCAL-01M36YFXAHPXMCFAGWQABT6KCE (GDN block-table row overflow).

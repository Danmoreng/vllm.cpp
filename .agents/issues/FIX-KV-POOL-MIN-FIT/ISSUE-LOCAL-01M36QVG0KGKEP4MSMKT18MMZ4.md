ID: ISSUE-LOCAL-01M36QVG0KGKEP4MSMKT18MMZ4
Title: engine: a request the hybrid KV pool can never hold waits forever instead of being refused
Row: FIX-KV-POOL-MIN-FIT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

On Qwen3.8-27B EXL3 with a DFlash2 draft (tree d4738d241, dgx:gpu0, leases 55f0238f and 68ae4d65), a server started with --max-model-len 16384 --num-blocks 1024 --max-num-seqs 1 accepted a 15,624-token chat request and never returned it. The client timed out at 300 s, the server stayed alive, and nvidia-smi GPU utilization read 0 for every sample, while a working request on the same box read nonzero. The same prompt completed in 26.9 s on a 17,408-block pool. An 8,305-token prompt completed on the 1,024-block pool. The startup check admitted the configuration: check_enough_kv_cache_memory sizes the pool from KVBytesPerBlock, which counts only the attention groups (kv_cache_interface.cpp:241-266). The scheduler's MambaManager in the default 'none' mode requests cdiv(num_tokens, block_size) blocks for the GDN group as well (single_type_kv_cache_manager.cpp:788-795), so one request needs about twice the blocks the check assumed. When allocate_slots cannot place a request, the waiting loop breaks without an error (scheduler.cpp:929-940), so the request waits forever. Expected: either refuse the configuration at startup, or reject the request with an error. The upstream vLLM comparison of the none-mode Mamba allocation is not yet made.

## Resolution

-

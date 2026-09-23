ID: ISSUE-LOCAL-01M36XJNF0TRNZBH7GYCW756AQ
Title: engine: the none-mode GDN KV group claims cdiv(len, block_size) + k pool blocks where vLLM claims 1 + k
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

MakeQwen3_5KVCacheSpec and the other hybrid registries build the mamba_cache_mode 'none' MambaSpec at the attention block size (32). vLLM sets the Mamba block size to max_model_len in 'none' mode (vllm/model_executor/models/config.py:645-657 @ e126687a9a), so its MambaManager claims cdiv(num_tokens + bs*k, bs) = 1 + k blocks per request. Ours runs the same arithmetic at bs = 32 and claims cdiv(num_tokens, 32) + k blocks from the shared pool. At max_model_len 32768 with a speculative config one request needs 3 * 1024 + k pool blocks here and 2 * 1024 + 1 + k at vLLM geometry. FIX-KV-POOL-MIN-FIT makes the startup check count the current claim; it does not change the claim. Moving the block size touches every hybrid registry and the runner's GDN block handling, and needs a developer decision.

## Resolution

-

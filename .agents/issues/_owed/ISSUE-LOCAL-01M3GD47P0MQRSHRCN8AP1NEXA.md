ID: ISSUE-LOCAL-01M3GD47P0MQRSHRCN8AP1NEXA
Title: Document the ABI 29 decision API and its model routing
Row: -
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: 2026-09-27

## Problem

The C API reference states ABI 26 and omits vllm_decide. The capability table includes Tev1 under vllm_decide although the adapter refuses that architecture. README news omits the new decision models.

## Resolution

27 September 2026: repaired in 49cdde49f325c2f4bc088de192cfc28f7ec9d777. Independent source review found no defects; focused CPU documentation checks, eight local links, two JSON bodies, and scratch mutations passed. Implementation and review preflights both retain 27 baseline/environment failures and 14 skips. No runtime behavior changed. See .agents/specs/docs-decision-api.md Outcome.

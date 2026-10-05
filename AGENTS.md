# AGENTS.md: local B70 development

This checkout is a direct collaboration with the developer. Stay on the current
branch. The developer closed the P0–P7 performance work period on 2026-10-05 after
the typed-memory speculative GDN SLM change, product commit `0ea34de4e`.
`docs/B70_EXL3_NATIVE_PERFORMANCE_IMPLEMENTATION_PLAN_7536aeded_EN.md` is now a
historical executed plan, with unresolved qualification gates preserved. Read
`docs/b70-exl3/PRO_PERFORMANCE_REVIEW_2026-10-05.txt` and `RECOVERY_STATUS.json`
for the final checkpoint and Pro handoff. Do not start another optimization or
restart P0–P7 until the developer supplies a new instruction or plan. The
developer's current instructions take priority. The recovery and migration
plans remain scope/evidence; archived EXL3/GPTQ plans are historical references.
GPTQ remains diagnostic only.

## How we work

- Use one assistant. Never start subagents or delegate implementation or review.
- Stay on the current branch. Do not create worktrees, branches, pull requests,
  or GitHub workflows unless the developer asks.
- Choose one small plan step, make the change, and run the smallest relevant
  focused test immediately. Report what changed, the exact result, and the next
  proposed step before moving to a larger step.
- Keep changes scoped to the step. Do not create issues, specs, claims, roadmap
  updates, or other process records automatically.
- Ask the developer when a missing fact changes behavior or when an action would
  affect a running service or discard work. Resolve routine code choices from
  the local source and the plan.

## Checks

- Do not run `agent-start`, `agent-role`, `agent-preflight`, `agent-ready`, full
  test suites, mutation reviews, CI, or broad verification gates unless the
  developer explicitly requests them.
- For a code change, run only a focused compile, unit test, or functional test
  that exercises that change. Report a test that cannot run; do not replace it
  with a claim of success. Documentation-only edits need no automatic test.
- Files under `.agents/` and repository check scripts are reference material,
  not mandatory workflow for this local branch.

## Local machine

- Use the local Intel Arc Pro B70 directly when the current focused step needs
  it. Do not use resource-controller leases or GPU lock files.
- The developer authorizes exclusive GPU use for this migration. Keep the
  production `b70-qwen38-vllm.service` stopped while working. No other vLLM
  instance may run on the card alongside the instance being built/tested.
  Focused GPU tests and isolated pinned-oracle captures are authorized; run
  oracle and native GPU work sequentially, and leave production stopped.
- Do not change drivers or power settings, install large dependencies,
  download model weights, or run long benchmarks without the developer's
  direction.
- Use vLLM source and the pinned model as behavior references where relevant.
  State any untested assumption or observed difference plainly.

## Git

- Do not push, merge, force-update a ref, or publish anything unless the
  developer asks. Commit a completed step when the developer asks.
- A later public pull request can use a clean branch containing the product
  changes. Create that branch only when the developer requests it.

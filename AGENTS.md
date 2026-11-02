# Agent instructions

Read `HANDOFF.md` before changing anything. It is the single source of truth for
the current state, known failures, and next actions.

## Hard rules

1. Do **not** commit `.runtime/`, `.venv/`, `artifacts/`, `results/`, logs, or
   any API key/password.
2. The current method is **CogniTune v2** (`cognitune_v2`), not the legacy
   five-expert `cognitune`. Do not revert to the legacy optimizer in new
   experiments.
3. The experiment protocol is: existing baselines are **not** rerun; only
   `cognitune_v2` and `mctuner` are run. Each once. MCTuner gets a 3-run median
   only if it sets a new best. Do not silently change this.
4. No LHS/Sobol/Halton/quasi-random initial design. Use the persisted shared
   IID initial design already in the repo/runtime.
5. Real PostgreSQL runs need `/root/GPTuner` (BenchBase + PostgreSQL 14.9) and
   `.runtime/cognitune.env`. Without them, use the synthetic evaluator only.
6. Run `pytest -q` before and after changes. Report failures, do not hide them.

## Current blocker

PostgreSQL storage I/O is degraded (82% disk use, ~1.6 MB/s fsync); startup and
checkpointer processes enter uninterruptible `D` state. Do not launch long real
runs until `pg_isready -h 127.0.0.1 -p 55432` is healthy and a 12-round probe
completes without `CRASH`.

## Start prompt for a fresh agent

> Read AGENTS.md and HANDOFF.md. Check git status, run `pytest -q`, and verify
> PostgreSQL health. Continue from the "Next actions" section of HANDOFF.md.
> Do not rerun old baselines, do not commit artifacts/secrets, and do not
> start a 50-round real run before the 12-round probe is clean.

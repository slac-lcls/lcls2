# Response to implementation-only review

Reviewed code: `f849540905a19cc2c18d620702088ade120740fe`.
Verdict: **pass for code only**, no blocking findings. Job 44009313 is queued;
there are no hardware results yet. This is not overall benchmark acceptance.
No benchmark code changed after review or job submission.

| ID | Response |
|---|---|
| SBT-01 | Optional cleanup deferred to future harness reuse, preserving queued code. README identifies old instrument.py as lineage-only provenance; the active wrapper is bulk_accounting.py. The inherited extra sys.path insertion is unused by current imports. |
| SBT-02 | Accepted as a required results-report obligation. audit.py already emits callback counts and subbatch histograms. README explicitly describes coupled read-size/ownership/execution-cut effects. Populate measured results and review them after the job completes; do not close overall acceptance before that. |
| SBT-03 | Optional cleanup deferred: exclusive allocation supplies more than eight physical cores; final auditor rejects fewer than eight. campaign.py never passes the inherited --preflight option (uses events=200), so the dead raw-only flag does not affect submitted commands. |

Coordinator checks: Python and shell syntax passed. An additional offline check
executed the frozen B CPU planner and exactly matched all independent predicted
histograms for 200 and 10,000 events at the three targets. This is saved in
cpu-plan-check.json and cpu-plan-check-command.txt, and is not GPU/IO evidence.
Reviewer had only Read/Glob/Grep, could not execute checks or read the frozen
runtime, and disclosed these limitations. Full result review remains pending.

Next: inspect Slurm completion; audit native output and run test_audit.py; record
actual rates, calls, sizes, subbatch histograms, NIC and cache observations;
freeze native evidence and submit a results-review round. If a required check
fails, diagnose it before reporting any performance result as accepted.

Workflow: [implement-review-loop](/sdf/home/m/monarin/.codex/skills/implement-review-loop/SKILL.md).

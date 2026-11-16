# Validation before independent review

- Controller tests: 16/16 passed (Python 3.11, 1.875 seconds). Synthetic offline
  adapters only; includes real local subprocess timeout, frozen artifact changes,
  phase gates, failed/missing samples, settings/event/cache mismatches, allocation
  changes, within/across-phase placement drift, complete AA/AB orchestration and
  four statistical classifications. No MPI, scheduler or GPU execution.
- Catalog checker: pass; 9 campaigns, 2,000 file hashes, 240 arithmetic cases.
- `git diff --check`: pass. Changes confined to maintained performance scripts,
  template/navigation and current task/plan notes. Frozen historical harnesses,
  runtime, public examples and build configuration unchanged.
- Candidate `prepare`: success with explicit qualification blockers; no execution.
  Frozen demonstration: /pscratch/sd/m/monarin/psana2-gpu/pr-regression-template-20261116/draft-check/.
- Run51 status: native Configure/standard calibration selection audit and complete
  402,653,184-byte status payload read; compact evidence and executed source retained.
- Native adapter/correctness, A/A, A/B and calibrated numerical qualification:
  NOT performed. Retention owner/cross-site identity/compute checks remain pending.

Commands from the source checkout:

```bash
python -B psana/psana/gpu/scripts/performance/regression/test_campaign.py -v
python -B psana/psana/gpu/scripts/performance/analysis/check_catalog.py
git diff --check
python -B psana/psana/gpu/scripts/performance/regression/campaign.py prepare \
  psana/psana/gpu/scripts/performance/regression/profile.json /new/scratch/draft
```

Tests used TMPDIR=/pscratch/sd/m/monarin/codex-tmp and Python at
/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python.
The status audit instead used the existing PR299 Python 3.9 native installation;
its selected source/runtime details are recorded with the audit scope.

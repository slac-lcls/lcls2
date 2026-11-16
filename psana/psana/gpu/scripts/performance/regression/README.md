# Maintained PR regression controller

This source-only template creates **new** frozen campaigns. It does not alter or
import execution code from the PR187/208 or PR299 archives. It implements profile
freezing, paired scheduling, process isolation/timeouts, receipt validation and
analysis. A campaign supplies a reviewed native adapter for its exact psana
revisions and site; there is no claim that a generic adapter works across changing
private runtime APIs. The supplied profile is deliberately not execution-ready.

Read [campaign requirements and agent instructions](../../../docs/performance/templates/pr-regression.md)
and the [dataset decision](../../../notes/pr_regression_template_20261116/README.md).

## Commands

Requires Python 3.9+ and its standard library. From the repository root:

```bash
python -B psana/psana/gpu/scripts/performance/regression/test_campaign.py -v
python -B psana/psana/gpu/scripts/performance/regression/campaign.py prepare \
  psana/psana/gpu/scripts/performance/regression/profile.json /new/scratch/campaign
```

`prepare` always refuses an existing destination. It copies supplied artifacts,
resolves artifact paths relative to the submitted JSON, records the exact profile
and controller digest, and generates all phase schedules plus explicit blockers.
For adapter_command use absolute executable/script paths and standalone
`{request}` and `{result}` tokens. There is no shell interpolation. Once prepared,
do not edit the profile or receipts; prepare a new directory after changes.

For a **qualified**, reviewed campaign inside its authorized compute allocation:

```bash
python -B /frozen/source/campaign.py execute /scratch/campaign correctness
python -B /frozen/source/campaign.py execute /scratch/campaign aa
python -B /frozen/source/campaign.py execute /scratch/campaign ab
python -B /frozen/source/campaign.py summarize /scratch/campaign ab
```

The controller must be byte-identical to the prepared version. It never submits
Slurm jobs. Every phase gets a new directory; interruption/failure is preserved
and cannot be silently resumed or overwritten. Correctness precedes A/A; passing
A/A precedes A/B. `profile` is a separate diagnostic phase after correctness and
never contributes to clean timing. Four/six/etc. even pair counts balance AB/BA
order within each cell. A/A runs the baseline installation on both sides.

Each native process receives request.json and writes result.json; stdout/stderr,
argv, exit code and external wall time remain alongside it. Native evidence must use distinct files inside its sample folder and is
hash-checked before acceptance. Full observed settings must equal the frozen
contract; qualification flags also require hashed receipt artifacts. Every configured cell/pair is required. Failures,
missing results, settings changes, placement changes, nonfinite timing and cache
violations reject the phase. Never filter out a slow sample after seeing timings.

`summary` output reports primary loop-time and secondary setup/first-batch/process
changes separately. Positive means candidate slower. It uses the mean of paired
percent changes and a two-sided 95% Student-t interval (rounded tabulated critical
values), with the profile's predeclared symmetric margin. Only an interval fully
inside that margin is equivalent; entirely above/below is regression/improvement;
overlap is inconclusive. The overall A/B label `not-equivalent` is **not** a claim
of regression: inspect each cell's outcome. Intervals are descriptive and assume
independent paired blocks; six pairs are a screen, not a universal performance
claim. Do not add independently maximized timing phases or pool sites/cells.

This controller checks a contract, not the truth of arbitrary adapter assertions.
The adapter code, numerical reference, native instrumentation and receipts need
independent review and hardware qualification. Synthetic CPU tests prove the
controller's failure handling; they are not native psana acceptance.

Read the [component performance policy](../../../docs/performance/templates/component-regression.md)
for reuse of existing unit/device tests and native scheduling counters. The six
raw cases stay unchanged. The controller's loop-equivalence gate is only one
part of campaign acceptance: qualified component bounds require adapter evidence
and results review; this controller does not yet collect or automatically gate
those counters. Do not treat `profile`'s single diagnostic pair as a statistical
component-timing comparison.

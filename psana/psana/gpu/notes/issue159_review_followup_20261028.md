# Issue159 / PR246: follow-up to Claude's review

October 28, 2026. The small fixes, tests and documentation requested after
[Claude's review](https://github.com/slac-lcls/lcls2/pull/246#pullrequestreview-5686620230)
are implemented and focused validation passed. Review baseline: `e7c700ba2`.
Review `e7c700ba2..HEAD` on `integrate/issue159-pr187-pr208` for this follow-up.
Claude found no blocking defect in cumulative Stages 1–4 or the dependency merge;
Mona requested these corrections before another review and further progression.

## Disposition of the six findings

1. **Copies accumulated across a slot:** keep the existing lifetime. A new CPU
   regression uses three distinct event roots and a shared budget in one actual
   `EventPool` retirement slot. Two 30-byte copies exhaust a 60-byte budget;
   the third access raises `GpuMemoryPressureError` with budget-sizing guidance,
   does not publish a field, and changes no existing charge. View-context exits
   leave the earlier charges live. `finish_retire_next()` releases them, and a
   later event can allocate again. CPU fixtures use logical bytes; existing CUDA
   coverage checks the pool's rounded backing charges. Document the absence of
   admission planning and automatic trim/retry for these lazy copies, including
   the current `.on_cpu` / `.on_gpu` intermediates.
2. **Host waits during retirement:** retain the current synchronization behavior.
   The performance plan now explicitly includes bulk-on unaligned fields with
   delayed consumers and separate retirement timing, plus bulk-off comparison.
   No deferred-release redesign or performance measurement is included here.
3. **Empty unaligned fields:** take the direct-view path when `nbytes == 0`, after
   all existing status/type/shape/byte-count/bounds validation. There are no typed
   payload loads, allocation charges, copy-ready events, or copy retention lists.
   The normal borrowed-context consumer event still exists. CPU coverage forbids
   allocation and event construction during empty-field materialization; CPU and
   device parameter grids assert that empty fields leave copy state untouched.
4. **Task descriptors:** both parser and user-kernel docs now require checking
   the actual `raw_ptr + offset` alignment before typed loads, with byte-safe
   loads as the fallback. Task descriptors still expose packed original input;
   public field-view copies do not change that contract. Keep the `on_gpu_view`
   caveat: first unaligned access may allocate and copy.
5. **Reporting:** clarify `_GpuMemStats.retained` documentation. Live private
   `field-copy` entries legitimately contribute to it. A new CPU regression runs
   for an IPC owner, importer, and private-constant owner, checking separation
   from `constants` / `borrowed`, charge release, and snapshot non-mutation while
   an admission hold is active. Existing MPI coverage remains the real-process
   evidence for those interactions.
6. **Inherited host scalar transport:** tracked separately as
   [issue #247](https://github.com/slac-lcls/lcls2/issues/247). No serializer or
   loader change is included. The previous MPI scalar claim remains limited to
   the GPU-manager input boundary.

## Validation

- **377 CPU tests passed**, no failures/errors/skips: field views, input-copy
  lifetime, input windows, retirement, input APIs, and memory reporting.
- Job **42254009**, **sdfampere010**, A100-SXM4-40GB, completed with **exit 0:0**
  in 18 seconds. **302 device field-view tests passed**, followed by **302 under
  compute-sanitizer memcheck**, with **zero sanitizer errors** and no skips.
- The CUDA grid includes all ten numeric types, scalar/array/empty shapes,
  shifted bases and offset residues, plus delayed cross-stream cache ordering,
  open-child retirement, ragged/absent fields and rounded accounting.
- Frozen-source/install hash checks passed before and after CPU and GPU runs
  (**930 files**). Every loaded GPU Python module resolved to the frozen new
  package; other psana modules and mapped project libraries resolved to the
  original verified native installation. Tests did not modify or relocate it.
- The device harness reports one `PytestUnknownMarkWarning` for the existing
  `gpu` marker because the standalone frozen bundle lacks marker registration.
  This did not skip or deselect tests; all 302 executed in each run.

This is focused validation of the review delta, not a new full native build or
MPI campaign. The runner explicitly selects a frozen GPU Python package while
reusing unchanged non-GPU/native code from:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159-target-merge-review-20261028-r1/trial/install`.
The new production behavior is only the empty-field guard in `gpu_input.py`;
`gpu_events.py` changes only its reporting docstring. Earlier MPI results still
refer to runtime `31c4c2f2`, not a newly measured head.

Artifacts, commands, source patch, environment, origin audits, JUnit reports and
hash manifest are preserved at:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/pr246-review-20261028-r1`.
[Committed evidence](evidence/issue159_review_followup_20261028.json) records the
result counts, exact changed-source hashes, scheduler state and provenance.

## Next review and gates

Claude should review this delta and confirm the six dispositions above. The
main fix branch remains unchanged and PR246 stays open. Do not merge or proceed
to Stage 5 before review and acceptance.

After follow-up review, run the matched cold and field-view screens. Preserve
frozen A `90c05965c`; the matched unfixed target remains `8cf05e86a`. The B candidate
must now include this empty-field correction, rather than reuse `31c4c2f2` as if
it were the latest runtime. Freeze and identify the reviewed B revision and
verify its separate build and dependencies before timing. No performance job
or Stage 5 implementation was started for this follow-up.

# Master synchronization: integration decisions

**November 25 status:** this is the original integration decision record.
The [current overview](current_status.md) includes the subsequent PR326 and
EndRun fixes, their passing verification/CI, and remaining review/landing work.
The scoped decisions and review limits below remain historical evidence.

This records the code decisions for [PR #331](https://github.com/slac-lcls/lcls2/pull/331),
the first master synchronization into `features/psana2-gpu` after #299.
Decisions were recorded November 18–19, 2026 and consolidated November 23.
The standing synchronization tracker is [#330](https://github.com/slac-lcls/lcls2/issues/330).

| Role | Revision |
| --- | --- |
| Pre-sync GPU baseline | `546764e0338225d026faa29627297a4a390d3396` |
| Incoming master | `785f7ef5dd2428bc0f1fb87c0ff1ebbab4d37935` |
| Reviewed implementation | `5bd5c7ba300cf0a0894b5b904d3975d5ba0b3cad` |
| Merge base | `f743ef5bef58033cdb9d38474f202ee92133994e` |

The implementation is a two-parent merge, GPU baseline first and master second.
The original documentation commits did not change that implementation.
Later PR326 and EndRun repairs are recorded in the current overview; the code
reviewed here remains `5bd5c7ba`. Preserve master ancestry when landing this PR:
use a merge commit, not squash or rebase. PR #326 was not part of this original
integration review; it is included in the later combined bundle.

An accepted integration decision means retain the selected code. It is not a
GitHub approval, a test pass, or full integration acceptance. Outstanding tests,
known failures and performance evidence are recorded in [validation.md](validation.md).

## Conflict resolutions

All five resolutions passed their scoped independent code reviews. Conflict 3
required two rounds; conflicts 1, 2, 4 and 5 passed together in one round.
Codex coordinated the work and runtime evidence; Claude independently inspected
the pinned clean source. The reviewer evaluated supplied runtime receipts rather
than executing those tests. No further production patch was required by these
review rounds.

| ID | Conflict | Decision and resulting behavior | Evidence and limits |
| --- | --- | --- | --- |
| C1 | `CHANGELOG.md` | Accept combined histories. Retain all incoming master entries and the GPU #298 entry, each once. | All 27 master entries retained; removing the complete GPU entry restores master's text. Inherited missing attribution remains a follow-up. |
| C2 | `psana/psana/psexp/ds_base.py` | Accept the union of parent options, including GPU options and `shmem_transitions_only`; keep its false default, true forwarding and unknown-key warnings. | Exact 41-key union checked; constructor/forwarding harness passed. Existing supervisor-key warnings were not changed. |
| C3 | `psana/psana/psexp/node.py` and callback-manager integration | Accept with an integration adapter: retain master's persistent `CallbackEventBuilderManager` and its two-part `batches()` interface; add `batches_with_gpu()` yielding `(cpu_batches, {}, step_batches)` for the GPU branch's three-part consumer. Close the callback stream in `finally`. Preserve MPI layout guards, callback statistics, transitions and destinations. | Six new adapter cases cover interface shape, trailing steps, exhaustion, early close and errors. Scoped review passed with completed byhand evidence. Callback-plus-GPU routing remains rejected; the deferred repeated-run MPI timeout is not resolved by this decision. |
| C4 | `psana/psana/tests/test_extract_subset_xtc2.py` | Accept comment reconciliation with executable behavior unchanged. | Executable AST matches both parents; all seven tests passed, including the `smdwriter` subprocess round trip. |
| C5 | Deletion of `setup_env.sh` | Follow master's deletion. Carry the GPU branch's pip CUDA-library discovery into `setup_env_ana.sh`; update primary README/build messages. | Actual SDF activation, library loading and CuPy RawKernel smoke passed without injected toolkit paths. System-toolkit-only and broader DAQ/hutch portability were not established. |

The candidate also preserves the union of curated psana CI test selections and
master's PR trigger, including the new callback-adapter test. CI selection alone
does not provide the full validation coverage in the companion report.

### Retained conflict-review follow-ups

- C3: realistic injected-stream fixture and direct constructor/close coverage;
  inherited duplicate/partial callback statistics; the existing 200-character
  XTC filename limit. Passing the adapter tests is not end-to-end cleanup proof.
- C5: a live GPU test README still references the removed setup script and must
  be corrected before following that recipe. A historical GPU-steps harness
  suppresses setup failure and must be migrated before reuse. Retired benchmark
  wrappers and DAQ/hutch examples need separate treatment; this PR does not
  certify those environments or restore retired benchmarks.
- The repeated-run callback MPI timeout is recorded separately in validation;
  it was nonblocking only for the scoped single-run review and later explicitly
  deferred by Mona. It was not proved inherited or fixed.

## Automatically merged existing-file changes

This inventory excludes additions, deletions, renames and the five conflict
groups above. Its stable item numbers group existing-file modifications by
behavior. Automatic textual merging was not used as proof of compatibility.

Mona accepted items 1–6 individually and accepted items 7–15 under the following
criterion: identify an API or behavior interaction that breaks the GPU branch,
or breaks incoming master functionality through GPU-branch changes. A new API,
unrelated detector behavior change or DAQ-private interface change alone does
not require an integration correction. No additional cross-branch break was
identified in this inventory. This was a source-level assessment, not an
exhaustive DAQ review or an additional independent review loop for all 15 groups.

| Item | Area | Integration decision | Rationale and retained boundary |
| --- | --- | --- | --- |
| 1 | Jungfrau calibration | Accept as-is | Reuse the first raw array and remove it from cached calibration-option comparisons; retain master's MPI-initialization removal and common-mode clarification. No calibration arithmetic change from raw-array reuse. |
| 2 | Shared detector-image assembly | Accept as-is; low direct GPU risk | CPU NumPy assembly caches indices/copy plans and avoids disabled debug formatting. Current GPU input/calibration uses panel arrays, not this image-assembly path. CPU/hybrid image consumers still need ordinary regression coverage. |
| 3 | Jungfrau geometry | Accept as-is; low direct GPU risk | Honor requested mask dtype and correct V2 pixel-size arrays. Default calibration and current GPU kernels do not use the changed geometry methods; explicit geometry-mask/area consumers can see corrected results. |
| 4 | XTC float16 support | Accept as-is | Append type 13 without renumbering existing types; retain two-byte sizing, C++ access and Python array conversion. Existing Jungfrau raw remains uint16. GPU float16 input parity is separate, tracked in [#349](https://github.com/slac-lcls/lcls2/issues/349). |
| 5 | Calibration lookup | Accept as-is | Keep latest selection for run 0/None and validity-range selection for positive runs. Preserve loaded experiment constants if detector-DB fallback returns None. No GPU-path adaptation needed. |
| 6 | Smalldata missing integers | Accept as-is | Use representable missing-value sentinels for narrow signed and unsigned dtypes, avoiding NumPy 2 overflow. Supplied values and GPU arithmetic are unchanged; downstream readers must recognize the missing-value representation. |
| 7 | Build/package integration | Accept as-is | `dgramCreate` implementation, build target and active repository callers migrate together to `psalg.utils.dgramCreate`. Site-specific build/import evidence does not certify external old imports or every DAQ/toolkit combination. |
| 8 | Existing CI workflows | Accept as-is | Retain PR triggers/cancellation and combined test selections. Manual release triggering changes automation policy. Selected CI is not full slow/GPU/performance coverage. |
| 9 | Shared-memory calibration prefetch | Accept as-is | Transition-only behavior is opt-in with defaults preserved; diagnostics do not change queue APIs. No ordinary file/MPI GPU incompatibility identified. Production shared-memory behavior is not certified. |
| 10 | Other detector/calibration behavior | Accept as-is | ePix/UHR and deployment changes are outside current Jungfrau GPU calibration; no cross-branch incompatibility identified. This does not promise identical ePix outputs or new GPU detector support. |
| 11 | Polar binning | Accept as-is | CPU in-ROI accumulation retains signature/output shape and is not called by the inspected GPU runtime/benchmarks. No measured speedup or new numerical-equivalence result claimed. |
| 12 | GPU DRP pipeline | Accept as-is | `psdaq/drpGpu` interfaces are DAQ-private, not psana GPU interfaces; those implementations match master. Shared format/build exposure is covered by items 4 and 7. No DAQ hardware-validation claim. |
| 13 | DAQ transport/event building | Accept as-is | No incompatible psana GPU header/descriptor assumption identified. Moving `keepRaw()` to the base class preserves its bit expression and the shared datagram layout in that diff. |
| 14 | DAQ scheduling/hardware/operations | Accept as-is | DAQ resource pairing and configuration do not replace psana GPU scheduling or change the reviewed recorded Jungfrau raw layout. Operational validation remains separate. |
| 15 | Examples/diagnostics/presentation | Accept as-is | Imports follow the package migration; inspected logging/GUI/diagnostic changes do not alter GPU data contracts. No additional incompatibility identified. |

### Details relevant to downstream users

- Item 1 does not imply one fewer disk read or array copy: it removes a repeated
  `raw()` call. CPU numerical references and performance still need validation.
- Item 3 leaves pixel coordinates and panel/data layout unchanged. Enabling
  geometric center masking can exercise the corrected mask implementation.
- Item 4 does not switch calibration arithmetic to half precision. GPU typed
  views lack type 13, and dense gather supports uint16/float32; unsupported views
  are rejected. Python rank-zero float16 materialization is also absent from
  the reviewed scalar switch, a separate master-side limitation.
- Item 5's bare `run=None` without time/version still hits the existing query
  assertion. The existing run-selection tests passed 8/8; that check was not a
  live calibration-service or GPU test.
- Item 6 uses signed minima for int8/int16 and unsigned maxima for unsigned
  integers. Signed int32/int64 retain `-99999`, floating types retain NaN.

## Provenance

The original working records are retained at the following SDF location; they
are not implied to be tracked files in this PR:

```text
/sdf/home/m/monarin/lcls2_worktree/psana2-gpu/notes/psana2_gpu_master_sync_20261117/
```

| Source record beneath that directory | SHA256 at consolidation |
| --- | --- |
| `existing_file_changes_ranked.md` | `e40b9fbf378d42a01b879004fa64e7ef264afbbe020fe789369cdf80eca656aa` |
| `review-conflict-3/round-02-review.json` | `78f119ae89e00c554ad9b2e423f6278c7b1e61f9371c1c091d0b0f286a0bdf3d` |
| `review-other-conflicts/round-01-review.json` | `ab1e64f4d444116978ed0d1d1ec73acf5cd9d869cac040d60b8f4eec650e61a1` |

Associated result/response files in those review directories retain stable
finding IDs and detailed dispositions. This document consolidates those
decisions; it does not introduce new code changes or new test results.

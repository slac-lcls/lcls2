# Issue159: final milestone review

Stages 1–7 are implemented, validated, reviewed and merged into the fix branch.
The final milestone brings the complete alignment fix from
`monarin/lcls2:fix/gpu-unaligned-reads-159` to
`slac-lcls/lcls2:features/psana2-gpu`. Final review and merge remain pending;
issue159 remains open.

On October 29, Mona requested fewer, larger PRs organized around meaningful
reviewable outcomes. Final reconciliation, documentation and any review fixes
belong to this one milestone PR. Do not create a separate PR for every stage,
note or wording correction. Intermediate work remains in the personal fork;
a material independent dependency change may still warrant a separate review.

## Target and candidate audit

- Latest fetched upstream target: `8cf05e86a7511f84a24f82b9c9720bc2308c471d`,
  unchanged from the finalized PR187/208 target used for validation.
- Candidate before these final documentation updates:
  `86a1eba3f3533cb9212d0643ad380906c1a7588d`, the merge of
  [personal-fork PR1](https://github.com/monarin/lcls2/pull/1).
- Upstream is an ancestor of the candidate. No new integration or conflict
  resolution is needed at this snapshot.
- Production GPU sources are identical to measured/reviewed B
  `405f868ed380c78b6009da3001f4535cb9ae647b`.
- All `psana/psana/tests/` files are identical to Stage 6's accepted
  `71f5c04a2` revision. Native sources, build inputs and dependency pins have no
  cumulative changes against the target.
- [Scope inventory](evidence/issue159_final_scope_20261029.json) records the
  exact comparison, every changed path and category counts before this final
  documentation commit. Later documentation does not change those code totals.

Claude's [Stage 7 review](https://github.com/monarin/lcls2/pull/1#pullrequestreview-5703561517)
found no blockers and independently reproduced the evidence. Both requested
wording corrections landed at `f64e31471` before the authorized merge. The
verifier passed again with byte-identical output. Stage 7's original review-
pending wording in the evidence JSON records its creation state, not a current
unresolved gate.

## Production review map

Packed XTC fields may have an odd address even inside an aligned allocation.
For example, a `uint16` field at `owner_base + 1` must not be read through an
unaligned `uint16*`. The fix tests the actual address, preserves valid aligned
loads, and provides a safe path for supported unaligned data.

| Module | Added / deleted LOC | Change to review |
|---|---:|---|
| `gpu_detector.py` | +27 / -3 | Actual-address alignment branch; byte reconstruction for uint16/float32 within the existing gather kernel; one thread per pixel, same aligned dense output |
| `gpu_input.py` | +117 / -11 | Lazy aligned field copies, cache/readiness events, root/child lease ownership, rounded budget accounting, failure/retry and safe retirement |
| `gpu_stream.py` | +2 / -1 | Pass the existing manager budget into input leases |
| `gpu_events.py` | +5 / -1 | Reporting docstring explains legitimate live field-copy residuals; no reporting algorithm change |
| **Total** | **+151 / -16** | Four production modules |

Dense gathering keeps its aligned typed-read fast path; unaligned reconstruction
preserves floating object bits and existing bounds/presence/zero-fill rules.
It adds no full-image input repack, XTC padding, extra gather launch or detector
type expansion. Dense gather still supports uint16 and float32.

`on_gpu_view()` continues to use the original parsed input. Aligned fields
remain borrowed direct views. The first unaligned access copies only that
field into aligned budgeted storage; cache hits reuse it. It does not switch
to the next dense-gather buffer. Readiness and consumer completion protect
source and copy lifetimes through input leases, including child view contexts.
Empty fields do not allocate alignment storage; absent segments remain absent.

The name `on_gpu_view()` describes borrowed access, not an unconditional
zero-copy promise. Existing per-event `.on_cpu` / `.on_gpu` APIs remain supported
through this safe field path. `.on_cpu` caches its host snapshot; `.on_gpu`
returns an independent device copy on each call. Their current copy counts and
memory costs are documented, not optimized or hidden in this milestone.
Routine detector processing should use batched tasks.

## Tests, tools and evidence

The cumulative diff also contains 16 test/fixture files (+1,474/-11), eight
benchmark/tool files (+641/-1), and two user guides (+83/-4). Test setup
corrections scope `PS_SMD_N_EVENTS` to the CLI/test that needs it and explicitly
request batch size 20 in the GPU task test; production defaults are unchanged.
Benchmark additions cover isolated gather, field access/retirement, input
leases, provenance and a benchmark-only shifted-input adapter. A private-stage
cache-reset helper and its tests preserve the historical cache investigation;
they do not add a production read-path dependency.

Most displayed PR lines are historical notes and machine-readable evidence:
31 evidence files (+63,520) and 28 notes (+4,801), before this final note/update.
They retain failed attempts, individual measurements, provenance and qualified
conclusions. Review production/test/user-doc changes first, then use the linked
reconciliation reports to inspect the relevant evidence. Preserve validation
history rather than removing it solely to reduce the displayed line count.

| Gate | Accepted evidence |
|---|---|
| Integrated correctness | [Stage 6](issue159_stage6_correctness_20261029.md): 1,646 core passed (includes 555 GPU integration cases), five byhand/MPI passed, five explicit real-Jungfrau cases passed |
| Focused device/sanitizer | Stage 6 maps PR246's 333 CUDA / 315 memcheck checks, the later 302 CUDA / 302 memcheck empty-field review checks, and ownership/budget tests; zero sanitizer errors on accepted runs |
| Copy API compatibility | [Stage 5](issue159_stage5_compatibility_20261029.md): 393 CPU, 60 CUDA and 60 memcheck passed |
| MPI ownership/reporting | [MPI report](issue159_mpi_reporting_20261028.md): 16 cases plus two real-data comparisons; later empty-field change has separate focused evidence |
| Cumulative aligned-data performance | [Stage 7](issue159_stage7_performance_20261029.md): same-node separate builds, cold private `/lscratch`, 4BD/1GPU, batch20, four balanced pairs per bulk mode; accepted mean loop +1.32% off / +1.95% on |
| Unaligned performance / field costs | Stage 7 preserves historical large-Jungfrau kernel/shifted-input screens and separate SDF/Perlmutter field/retirement results, without pooling sites or converting accessor timings to application throughput |

Counts overlap and must not be summed into a unique-test total. Stage 6 records
five non-GPU skips and seven default slow deselections; five real-Jungfrau cases
were run explicitly, two unrelated slow tests remain unrun. The first-attempt
shared-memory no-event failure did not repeat and has no established root cause;
the calibration HTTP connection failure also passed on retry. Accepted Stage 5
memcheck uses device-only backtraces after diagnosing host-backtrace retention;
no assertion or memory check was suppressed. These qualifications remain visible.

No new CPU/GPU/performance campaign is justified by this final documentation
step. Source/test equivalence establishes reuse of the accepted checks; this is
not a claim that a new final build was measured. Reassess if review changes
runtime, dependencies, gather, retirement or admission behavior.

## Accepted follow-ups and final gate

- [Upstream #256](https://github.com/slac-lcls/lcls2/issues/256): retained-copy
  retirement waits, slot-wide copy accumulation and admission planning. The
  delayed-consumer microbench shows about 7.1 ms of host waiting; that is not
  an image-copy kernel duration.
- [Upstream #257](https://github.com/slac-lcls/lcls2/issues/257): per-event API
  policy and optional direct-D2H/single-copy optimizations.
- [Upstream #247](https://github.com/slac-lcls/lcls2/issues/247): inherited host
  calibration scalar-shape transport, separate from scalar field correctness.

Review the cumulative four-module fix, test coverage, API costs, evidence reuse
and these accepted limitations. Verify the target has not moved materially
before merge. Final merge needs Mona's authorization; only after that merge
should issue159 be closed. No automatic closure is assumed for a PR targeting
a non-default feature branch.

## Final review follow-up

Claude reviewed PR262 at `f74ba6ab0` against unchanged target `8cf05e86a` and
found no blockers. Codex independently rechecked production/test equivalence
and the cumulative path scope. Both non-blocking documentation suggestions
were addressed on the same PR: the October 19–22 cold-rate history moved from
`docs/performance/read_staging.md` to the existing cold-rate investigation note,
and the personal-fork handoff now names PR262 as the current merge gate.
The read-staging guide is now identical to the target; user-documentation totals
above reflect that removal. The scope JSON remains the explicitly dated
pre-final-documentation inventory. No runtime or tests changed, so existing
validation remains applicable; no new test run is claimed.

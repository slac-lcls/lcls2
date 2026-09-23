# Stage 4 integrated bulk performance comparison

Job 38904896, sdfampere001, GPU-503baa09-51e4-3570-bd4e-02c202b862ff.

12 clean samples and three CPU-reference preflights passed. 3/3 separate diagnostics complete.
Timing audit: PASS. Full campaign: COMPLETE.

B++ = frozen standalone zero-wrapper baseline; Integrated-off/on = B++ plus ownership fixes and multi-owner bulk support. JF only.

| Cache | Variant | n | Median s | Events/s | Range events/s | Input GB/s | Range GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| cold | B++ | 2 | 70.996 | 140.9 | 140.7–141.0 | 4.727 | 4.721–4.732 |
| cold | Integrated-off | 2 | 73.208 | 136.6 | 135.8–137.4 | 4.584 | 4.557–4.611 |
| cold | Integrated-on | 2 | 111.328 | 89.8 | 88.8–90.9 | 3.014 | 2.978–3.051 |
| warm | B++ | 2 | 27.326 | 366.0 | 350.6–382.8 | 12.280 | 11.764–12.844 |
| warm | Integrated-off | 2 | 34.126 | 293.0 | 256.6–341.6 | 9.833 | 8.610–11.463 |
| warm | Integrated-on | 2 | 33.598 | 297.6 | 284.5–312.0 | 9.988 | 9.547–10.471 |

Rates use events or useful input bytes divided by median loop time.

| Cache | Comparison | Median paired throughput change | Range |
|---|---|---:|---:|
| cold | B++ → Integrated-off | -3.02% | -3.49%–-2.55% |
| cold | Integrated-off → Integrated-on | -34.24% | -34.64%–-33.84% |
| warm | B++ → Integrated-off | -17.77% | -32.97%–-2.56% |
| warm | Integrated-off → Integrated-on | +2.46% | -16.71%–+21.62% |

Paired changes compare the same repetition/cache mode. Two repetitions reverse variant order; not all positions can be balanced. Treat small differences as descriptive.
B++ preserves historical native binaries, whose hashes differ from the integrated build. B++ versus integrated off compares complete builds; a gap alone does not isolate Python integration overhead.
Cold means verified Linux-page-cache-cold local NVMe. KvikIO uses CPU fallback.
Clean timing includes event-loop setup and final GPU synchronization; staging, cache preparation, and teardown are excluded.
Allocation tracing and profiling are disabled in clean samples. These measurements do not validate true GDS or multi-BD performance.

## Separate diagnostics

| Variant | Access | Ledger+held peak MiB | CuPy used peak MiB | CuPy used end MiB | Reads |
|---|---|---:|---:|---:|---:|
| B++ | pixels | 2561.8 | 3201.8 | 2561.8 | 50000 |
| Integrated-off | pixels | 3201.8 | 3201.8 | 0.0 | 50000 |
| Integrated-on | pixels | 3201.8 | 3201.8 | 0.0 | 2885 |

## Diagnostic launch counts

| Variant | Parsed windows | Walk | Locator init | Grouped locate | Individual locate | Canonical gather | Eager wrappers |
|---|---:|---:|---:|---:|---:|---:|---:|
| B++ | 500 | 500 | 500 | 500 | n/a | 500 | 0 |
| Integrated-off | 500 | 500 | 500 | 500 | n/a | 500 | 0 |
| Integrated-on | 500 | 500 | 500 | 500 | n/a | 500 | 0 |

These counters cover named parser/gather kernels, not all CUDA launches. Missing factories are shown as n/a.

Diagnostic timings are excluded from throughput. Memory peaks are sampled; pool total/free and device residency differ from live owned bytes.

## Workload and artifacts

JF only, mfx101210926 run 387, streams 5–9; 10,000 events per sample,
batch 20, depth 1, 8 GiB budget, one BD plus SMD0/EB. No automatic/user D2H
in clean timing. KvikIO CPU fallback, eight I/O threads, 1 MiB tasks.
Each fresh MPI sample follows a 100-event compilation warmup.

Frozen B++ is the historical Z installation from job 38676923, verified against
all 25 recorded hashes. Integrated off/on use runtime ac87a93b2, validated
through Stage 3 without additional production changes.

Artifacts, frozen installs, scripts, and source/native manifests:
`/sdf/scratch/users/m/monarin/gpu-validation/bpp-integration-performance-20260923/`.
The `job-38904896/` subdirectory contains results.json, summary.json, profiles,
cache/placement measurements, and provenance.json. Job completed successfully
in 1h04m23s. Discarded job 38904093 contributes no samples.

See the [optimization checkpoint](../bulk_batched_optimization_baseline.md) for
functional coverage, scope boundaries, and follow-up priorities.

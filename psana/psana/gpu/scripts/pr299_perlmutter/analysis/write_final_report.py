"""Generate the durable report from completed, independently reviewed campaign evidence."""
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).resolve().parent
T = Path('/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6')
R = T/'results/campaign-job-61375111'
summary = json.loads((D/'campaign-summary.json').read_text())
validation = json.loads((D/'campaign-validation.json').read_text())
storage = json.loads((D/'campaign-storage-progress.json').read_text())
assert len(storage) == 48
assert 'COMPLETED' in (D/'campaign-slurm-accounting.txt').read_text()
cases = {r['case']:r for r in summary['results']}
samples = [json.loads(f.read_text()) for f in sorted(R.glob('timed-*/sample.json'))]
order = ['r51-gpu-events-bd1','r51-gpu-events-bd2','r51-cpu-events-bd1','r51-cpu-events-bd2','r51-cpu-steps-bd1','r51-cpu-steps-bd2']

def job_path(path):
    return '/job_61375111/' in path and '/step_' not in path

job_peaks = [value for s in storage.values() for path,value in s['memory_peak_bytes_by_path'].items() if job_path(path)]
job_events = {label:{path:value for path,value in s['memory_events'].items() if job_path(path)} for label,s in storage.items()}
assert all(job_events.values())
nonzero = {label:{key:value for counters in paths.values() for key,value in counters.items() if value} for label,paths in job_events.items()}
nonzero = {key:value for key,value in nonzero.items() if value}
periods = [s['observed_monitor_period_seconds'] for s in storage.values()]
errors = sorted({path for s in storage.values() for path in s['counter_read_error_paths']})
reclaim = {}
for key in ['pgscan','pgsteal','workingset_refault_file']:
    values = [s['job_memory_stat'][key]['last']-s['job_memory_stat'][key]['first'] for s in storage.values() if key in s['job_memory_stat']]
    reclaim[key] = dict(samples_observed=len(values),minimum_delta=min(values),maximum_delta=max(values),sum_sample_deltas=sum(values))
inventory = json.loads((R/'storage-inventory.json').read_text())
with (R/'timed-r51-gpu-events-bd1-pair0-0/storage.jsonl').open() as stream:
    first_storage = json.loads(next(stream))
memory_limits = {path:int(value) for path,value in first_storage['values'].items() if job_path(path) and path.endswith('memory.max') and value.strip() != 'max'}
rollup = dict(samples=48,job_memory_peak_bytes=max(job_peaks),job_memory_events_nonzero=nonzero,monitor_period_seconds=dict(minimum=min(p['min'] for p in periods),maximum=max(p['max'] for p in periods),mean_of_sample_means=statistics.mean(p['mean'] for p in periods)),reclaim=reclaim,counter_read_error_paths=errors,lustre_stats_observable=inventory['lustre_stats_observable'],storage_samples=sum(s['storage_samples'] for s in storage.values()))
rollup['job_memory_limit_bytes_at_first_gpu_sample'] = memory_limits
(D/'campaign-storage-rollup.json').write_text(json.dumps(rollup,indent=2)+'\n')

lines = [
    'PR299 Perlmutter Run51 performance screen — completed',
    '',
    'Execution and workload validation passed: 12 preflights and all 48 timed',
    'samples completed. The performance result is inconclusive for small PR',
    'effects: paired medians are near zero, with order sensitivity and temporal',
    'drift. No consistent regression across the six measured cases was observed.',
    'This is the reviewed reduced Run51 scope, not the full eight-case suite.',
    'Run77 remains deferred. The known C1 CPU Step-parent lifetime blocker is',
    'unfixed; these timings do not provide PR merge approval.',
    '',
    'Job 61375111: COMPLETED, exit 0:0, elapsed 02:44:14, nid003260.',
    'Account lcls_g, GPU premium QoS; requested walltime 04:03:00.',
    'Each workload step used one physical A100-SXM4-40GB. Slurm allocated and',
    'accounted for a full node (4 GPUs, 128 CPUs); see campaign-slurm-accounting.txt.',
    'BD1/BD2 denote one/two bigdata workers sharing that physical GPU.',
    '',
    'Paired throughput changes (positive means faster PR head)',
    'Each case has four complete pairs; order alternates B/H, H/B, B/H, H/B.',
    'Change = 100 * (head events/s / baseline events/s - 1). Ranges are the',
    'four observed pair values, not confidence intervals or equivalence bounds.',
    'The paired median is the median of four ratios, not a ratio of median times.',
    '',
    'Case                 Median change    Pair range             Loop median B/H (s)',
]
for name in order:
    c=cases[name];lo,hi=c['paired_throughput_range_percent']
    lines.append(f"{name[4:]:20s} {c['paired_throughput_median_percent']:+7.2f}%       {lo:+6.2f}% to {hi:+6.2f}%      {c['base_loop_median_s']:7.2f} / {c['head_loop_median_s']:7.2f}")
lines += [
    '',
    'Startup and whole-process measurements, separate from the event loop:',
    'Setup below is the median of each sample\'s maximum rank setup_seconds.',
    'These quantities have different scopes; do not add their medians together.',
    'Case                 Setup median B/H (s)    MPI wall median B/H (s)',
]
for name in order:
    c=cases[name]
    setup=[statistics.median(max(r['setup_seconds'] for r in s['rows']) for s in samples if s['case']==name and s['runtime']==v) for v in ('base','head')]
    lines.append(f"{name[4:]:20s} {setup[0]:7.2f} / {setup[1]:7.2f}       {c['base_process_wall_median_s']:7.2f} / {c['head_process_wall_median_s']:7.2f}")
lines += [
    '',
    'Interpretation',
    'GPU BD1 has an order reversal: median head change is +2.52% when second',
    'and -2.79% when first, a 5.31 percentage-point gap. Its overall paired',
    'median is +0.005%. GPU BD2 has a +0.98% paired median, but both variants',
    'slow in the final pair. CPU events timings also drift over execution order;',
    'the first CPU BD2 pair (+6.82%) is much larger than the other three pairs.',
    'Four pairs and uncontrolled cache/storage conditions cannot isolate these',
    'small changes as PR effects. No formal performance pass threshold or',
    'statistical equivalence claim is inferred. CPU controls use their existing',
    'iterator workload; these are not equal-byte CPU-versus-GPU speedups.',
    'plots/event-loop-order.png and .pdf retain individual timings in order.',
    'plots/paired-timings.csv retains all paired loop and MPI-wall measurements.',
    '',
    'Sustained throughput and I/O',
    'The primary GPU BD1 samples all completed 10,000 events. Last-half rates',
    'span 66.58–69.17 events/s; per-sample min/max rate ratios across the last',
    'three quarters span 0.923–0.977. There was no timeout or incomplete tail.',
    'GPU BD2 last-half per-BD rates span 43.36–51.13 events/s, with quarter',
    'min/max ratios 0.767–0.941. These rank-local windows differ and must not',
    'be added as if they shared endpoints; this variation further limits a',
    'small-effect conclusion. Actual logged endpoints are retained in',
    'campaign-validation.json; BD2 can have an unlogged tail below 100 events.',
    'Every timed GPU sample has 335,571,760,000 cumulative completed bytes, identical',
    'across paired variants. The 16 timed GPU samples total 5,369,148,160,000',
    'completed bytes. Counts sum disjoint pre-reset windows plus final tails;',
    'they are not unique dataset bytes, physical storage traffic, or isolated',
    'I/O bandwidth. KvikIO compatibility mode and CPU-fallback were verified.',
    'Sampled device memory peaks were 1,831 MiB for BD1 and 3,657 MiB for BD2;',
    'these 1 Hz observations are not exact allocation high-water marks.',
    '',
    'Storage and pressure evidence',
    f"Scanned {rollup['storage_samples']} storage records from all 48 timed samples.",
    f"Peak observed job-cgroup memory.current: {max(job_peaks)/1e9:.3f} GB (decimal).",
    f"Recorded job-cgroup limit at first GPU sample: {min(memory_limits.values())/1e9:.3f} GB.",
    ('All per-sample job memory.events deltas were zero, including high/max/OOM/kill.' if not nonzero else f"Nonzero per-sample job memory.events deltas: {json.dumps(nonzero,sort_keys=True)}."),
    f"Actual monitor period: {rollup['monitor_period_seconds']['minimum']:.3f}–{rollup['monitor_period_seconds']['maximum']:.3f} s; mean of sample means {rollup['monitor_period_seconds']['mean_of_sample_means']:.3f} s.",
    'Reclaim/refault counters are retained in campaign-storage-rollup.json.',
    'Lustre llite statistics and memory PSI were unavailable. Missing paths and',
    'read errors are recorded explicitly. Node/CXI/network counters are not',
    'process-attributed and network traffic need not be Lustre traffic. There',
    'was no cache eviction or staging, and no cold-cache claim. These counters',
    'cannot identify the cause of timing drift or isolate storage from reclaim.',
    '',
    'Scope, validation, and provenance',
    'Baseline: 11e4a13b541e2aefbfcb7168ab4d2040e2bcd4ba',
    'PR head:  acfd029a8d27898b5f165e02facd3c3bbc9c4ad2',
    f"Reviewed manifest: {summary['manifest_sha256']}",
    'Both pinned sources were built from empty local scratch build/install',
    'directories. Release parity: optimization=3, debug=false, b_lto=true,',
    'b_ndebug=true, cpp_std=c++20, build_daq=false. The frozen-source, harness,',
    'install, input metadata/hash and build-option checks passed again after',
    'campaign completion. Actual Python/native import prefixes were validated',
    'on compute ranks before and after workload execution.',
    'Local environment: Python 3.9.20, GCC 14.3.0, CuPy 13.6.0, KvikIO 24.08.02,',
    'CUDA runtime 12.9; mpi4py mapped Cray MPICH 9.0.1 (default module 9.1.0).',
    'Run51: all ten SMD hashes and bigdata sizes matched; 17,872 events available,',
    'first 10,000 selected, one step and no EndRun. Full payloads were not hashed.',
    'All 48 timed samples have the same exact independently referenced event',
    'timestamps; per-rank ordering and nested step assignments passed.',
    'All 24 pairs have identical physical-core/GPU placement; every BD was active.',
    'CPU roles remained GPU-uninitialized. GPU preflights checked raw input.',
    f"GPU UUID: {validation['gpu_uuids'][0]}",
    'GPU settings: one stream per BD, batch 20, 8 GB budget per BD, bulk reads,',
    '1 MiB bulk target; KvikIO compatibility ON, four threads, 1 MiB task size.',
    'Raw Jungfrau minimal task; calibration skipped; no full-image D2H or GDS.',
    '',
    'Readiness and independent review',
    '94 offline harness tests passed. Successful bounded pilot 61372763 and',
    'six-case sizing pilot 61373624 established sustained local measurements.',
    'The latter completed all six 10,000-event baseline cases and 12 preflights.',
    'Per-case symmetric timeout bounds and allocation time were derived from',
    'those completed measurements, rather than extrapolating a 120-event prefix.',
    'Independent Claude campaign review passed before submission, bound to the',
    'exact manifest and completed sizing evidence digest:',
    'b08dd6438ab7ffc5c24092555e205555119aa67afdbb3400c0a9773d3567518b.',
    'Readiness approval is in reviews/campaign-r1/review.json. It covers the',
    'campaign preparation; this final statistical interpretation is separate.',
    'Prior failed startup attempts and all review responses remain archived.',
    'Optional follow-ups remain documented: strict SLURM_JOB_END_TIME reliance',
    'and mocked execute(campaign) loop coverage before a later campaign retry.',
    '',
    'Coverage gaps and retained evidence',
    'Run77 was not found in the checked local locations; its head-only events',
    'versus nested-steps comparison and multi-step/EndRun coverage are deferred.',
    'No transfer or substitute dataset was used. C1 still affects a returned',
    'CPU Step after its temporary run.steps() parent is released. Nested-loop',
    'timings do not exercise or fix that contract.',
    f'Raw evidence and fresh builds: {T}',
    'Durable compact evidence: snapshot/campaign-job-61375111, with per-file',
    'scratch hashes in scratch-source-index.json. Large worker/storage logs',
    'remain on scratch. This directory contains the report, summaries, plots,',
    'CSV, analysis scripts, review records, and SHA256SUMS.',
    'The authorized reduced campaign and analysis are complete. No further',
    'jobs or external PR submission are needed for this completed screen.',
]
(D/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(rollup,indent=2))
print('Wrote FINAL_REPORT.txt')

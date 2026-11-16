"""Read-only, standard-library audit of the compact PR299 evidence package."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


SCRIPTS = Path('psana/psana/gpu/scripts/pr299_perlmutter')
EVIDENCE = Path('psana/psana/gpu/notes/evidence/pr299_perlmutter_20261115')
REPORT = Path('psana/psana/gpu/notes/pr299_perlmutter_20261115.md')
BASE = '11e4a13b541e2aefbfcb7168ab4d2040e2bcd4ba'
HEAD = 'acfd029a8d27898b5f165e02facd3c3bbc9c4ad2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums(path):
    entries = {}
    for line in path.read_text().splitlines():
        checksum, name = line.split('  ', 1)
        relative = Path(name)
        require(re.fullmatch('[0-9a-f]{64}', checksum), 'Invalid SHA256')
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe checksum path')
        require(name not in entries, 'Duplicate checksum path')
        entries[name] = checksum
    return entries


def near(actual, expected, context):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), context)


def package_files(repository):
    return {path for directory in (SCRIPTS, EVIDENCE)
            for path in (repository / directory).rglob('*') if path.is_file()} | {repository / REPORT}


def verify(repository):
    repository = repository.resolve()
    evidence = repository / EVIDENCE
    checksums = read_checksums(evidence / 'SHA256SUMS')
    files = package_files(repository) - {evidence / 'SHA256SUMS'}
    require(set(checksums) == {str(path.relative_to(repository)) for path in files}, 'Checksum coverage differs')
    for name, expected in checksums.items():
        path = repository / name
        require(path.resolve().is_relative_to(repository) and not path.is_symlink(), 'Unsafe package path')
        require(sha256(path) == expected, 'Checksum mismatch: ' + name)
    original = read_checksums(evidence / 'SOURCE_SHA256SUMS')
    mapping = load(evidence / 'archive-map.json')['files']
    for name, entry in mapping.items():
        require(name in checksums and checksums[name] == entry['sha256'], 'Mapped hash mismatch: ' + name)
        require((repository / name).stat().st_size == entry['bytes'], 'Mapped size mismatch: ' + name)
        if entry['source'] != 'SHA256SUMS':
            require(original[entry['source']] == entry['sha256'], 'Source hash mismatch: ' + name)
    manifest = load(evidence / 'manifest.json')
    approval = load(evidence / 'review-approval-campaign.json')
    review = load(evidence / 'reviews/campaign-r1.json')
    require((manifest['base'], manifest['head']) == (BASE, HEAD), 'Wrong measured refs')
    require(sha256(evidence / 'manifest.json') == approval['manifest_sha256'] == review['reviewed_manifest_sha256'], 'Approval binding differs')
    require(review['reviewed_commit'] == HEAD and review['verdict'] == 'pass', 'Historical readiness review differs')
    require(not any(finding['blocking'] for finding in review['findings']), 'Historical readiness blocker')
    harness = repository / SCRIPTS / 'harness'
    require({str(path.relative_to(harness)): sha256(path) for path in harness.rglob('*') if path.is_file()} == manifest['harness'], 'Frozen harness differs')
    require(sha256(evidence / 'reference-51.json') == manifest['reference_sha256'], 'Reference hash differs')
    sizing_index = load(evidence / 'provenance/sizing-source-index.json')['files']
    sizing_payload = dict(estimate=sha256(evidence / 'pilot-estimate.json'), results={name: entry['sha256'] for name, entry in sizing_index.items()})
    sizing_digest = hashlib.sha256(json.dumps(sizing_payload, sort_keys=True).encode()).hexdigest()
    require(sizing_digest == approval['pilot_evidence_sha256'] == review['pilot_evidence_sha256'], 'Sizing index/approval differ')
    build_options = []
    for variant in ('base', 'head'):
        options = {entry['name']: entry['value'] for entry in load(evidence / f'provenance/build-options-{variant}.json')}
        require(Path(options['prefix']).name == 'install-' + variant, 'Wrong build prefix')
        for name, value in manifest['build_options'].items():
            require(options[name] == value, 'Wrong build option: ' + name)
        build_options.append({name: value for name, value in options.items() if name not in ('prefix', 'python.platlibdir', 'python.purelibdir')})
    require(build_options[0] == build_options[1], 'Build option parity differs')
    reference = load(evidence / 'reference-51.json')
    job = evidence / 'campaign-job-61375111'
    status = load(job / 'status.json')
    require(status['complete'] and status['paired_samples'] == 24 and status['manifest_sha256'] == approval['manifest_sha256'], 'Incomplete/wrong campaign')
    samples = {path.parent.name: load(path) for path in job.glob('*/sample-summary.json')}
    require(len(samples) == 60 and sum(not sample['preflight'] for sample in samples.values()) == 48, 'Wrong sample counts')
    validation = load(evidence / 'campaign-validation.json')
    byte_totals = []
    for label, sample in samples.items():
        count = 120 if sample['preflight'] else 10000
        timestamp_digest = hashlib.sha256(json.dumps([entry[0] for entry in reference['events'][:count]], separators=(',', ':')).encode()).hexdigest()
        require(sample['label'] == label and sample['complete'] and sample['events'] == count, 'Incomplete sample: ' + label)
        require(sample['timestamp_sha256'] == timestamp_digest, 'Wrong event digest: ' + label)
        process = load(job / label / 'process-status.json')
        require(process['complete'] and process['mpi_process_wall_seconds'] == sample['mpi_process_wall_seconds'], 'Process status differs')
        require(sample['n_gpu_streams'] == 1 and sample['batch_size'] == 20 and sample['gpu_memory_budget_gb'] == 8 and sample['gpu_bulk_read'], 'Wrong workload settings')
        rows = sample['rows']
        require(len(rows) == int(sample['case'][-1]) + 2, 'Wrong rank count')
        require(sum(row['event_count'] for row in rows) == count, 'Rank event sum differs')
        near(sample['loop_seconds'], max(row['loop_seconds'] for row in rows), 'Loop maximum differs')
        near(sample['events_per_second'], count / sample['loop_seconds'], 'Rate differs')
        used_cores = set()
        for rank, row in enumerate(rows):
            role = 'smd0' if rank == 0 else 'eb' if rank == 1 else 'bd'
            require(row['rank'] == row['local_rank'] == rank and row['role'] == role, 'Rank/role mismatch')
            require(row['event_count'] > 0 if role == 'bd' else row['event_count'] == 0, 'Idle BD or active non-BD')
            cores = set(map(tuple, row['physical_cores']))
            require(len(cores) == 8 and not used_cores.intersection(cores), 'Invalid core affinity')
            used_cores.update(cores)
            expected_gpu = sample['mode'] == 'gpu' and role == 'bd'
            before = load(job / label / f'runtime-rank-{rank}.json')
            require(before['gpu_initialized'] == row['runtime_provenance']['gpu_initialized'] == expected_gpu, 'GPU role isolation differs')
            require(row['gpu_uuid'] == (validation['gpu_uuids'][0] if expected_gpu else None), 'GPU placement differs')
            provenance = row['provenance']
            install = Path(provenance['install'])
            require(install.name == 'install-' + sample['runtime'], 'Wrong runtime install')
            for origin in list(provenance['python'].values()) + provenance['native_libraries']:
                require(Path(origin).is_relative_to(install), 'Cross-prefix import')
            for origin, expected in provenance['native_sha256'].items():
                require(manifest['installs'][sample['runtime']][str(Path(origin).relative_to(install))] == expected, 'Loaded library hash differs')
            if expected_gpu:
                accounting = row['cumulative_io']
                require(accounting['compat_mode'] and accounting['bulk_read'] and accounting['io_path'] == 'CPU-fallback', 'Unexpected I/O path')
                if sample['preflight']:
                    require(row['input_checks'] > 0, 'Missing raw-input preflight')
        if sample['mode'] == 'gpu' and not sample['preflight']:
            byte_totals.append(sum(row['cumulative_io']['total_bytes'] for row in rows if row.get('cumulative_io')))
    require(len(byte_totals) == 16 and set(byte_totals) == {335571760000} and sum(byte_totals) == validation['gpu_timed_total_completed_bytes'], 'Completed I/O total differs')
    actual_order = re.findall(r'^PASS ([^:]+):', (evidence / 'logs/slurm-61375111.out').read_text(), re.MULTILINE)
    require(len(actual_order) == 60 and set(actual_order) == set(samples), 'Execution log coverage differs')
    with (evidence / 'plots/paired-timings.csv').open() as stream:
        pairs = list(csv.DictReader(stream))
    require(len(pairs) == 24 and len({(pair['case'], pair['round']) for pair in pairs}) == 24, 'Missing/duplicate pairs')
    changes = {}
    for pair in pairs:
        case, round_number = pair['case'], int(pair['round'])
        base = samples[f'timed-{case}-pair{round_number}-0']
        head = samples[f'timed-{case}-pair{round_number}-1']
        require(base['runtime'] == 'base' and head['runtime'] == 'head', 'Swapped runtime labels')
        require(actual_order.index(head['label']) - actual_order.index(base['label']) == (1 if round_number % 2 == 0 else -1), 'Wrong actual pair order')
        require(pair['order'] == ('base/head' if round_number % 2 == 0 else 'head/base'), 'CSV order differs')
        for left, right in zip(base['rows'], head['rows']):
            require(all(left[key] == right[key] for key in ('rank', 'role', 'physical_cores', 'affinity', 'gpu_uuid')), 'Paired topology differs')
        change = 100 * (base['loop_seconds'] / head['loop_seconds'] - 1)
        changes.setdefault(case, []).append(change)
        for key, expected in dict(base_loop_s=base['loop_seconds'], head_loop_s=head['loop_seconds'], base_mpi_wall_s=base['mpi_process_wall_seconds'], head_mpi_wall_s=head['mpi_process_wall_seconds'], head_throughput_change_percent=change).items():
            near(float(pair[key]), expected, 'CSV value differs: ' + key)
    for result in load(evidence / 'campaign-summary.json')['results']:
        ratios = changes[result['case']]
        require(len(ratios) == 4, 'Wrong pairs per case')
        near(statistics.median(ratios), result['paired_throughput_median_percent'], 'Median differs')
        near(min(ratios), result['paired_throughput_range_percent'][0], 'Minimum differs')
        near(max(ratios), result['paired_throughput_range_percent'][1], 'Maximum differs')
    for entries in validation['sustained'].values():
        for entry in entries:
            progress = [json.loads(line) for line in (job / entry['label'] / f"progress-rank-{entry['rank']}.jsonl").read_text().splitlines()]
            count = next(row['event_count'] for row in samples[entry['label']]['rows'] if row['rank'] == entry['rank'])
            require(0 <= count - progress[-1]['events'] < 100, 'Missing progress tail')
            require(all(right['events'] - left['events'] == 100 and right['seconds'] > left['seconds'] for left, right in zip(progress, progress[1:])), 'Invalid progress order')
            by_count = {point['events']: point for point in progress}
            for window in [entry['late_half']] + entry['quarters_after_first']:
                first, last = by_count[window['first_event']], by_count[window['last_event']]
                elapsed = last['seconds'] - first['seconds']
                near(elapsed, window['seconds'], 'Sustained elapsed differs')
                near((last['events'] - first['events']) / elapsed, window['events_per_second'], 'Sustained rate differs')
            rates = [window['events_per_second'] for window in entry['quarters_after_first']]
            near(min(rates) / max(rates), entry['quarter_min_max_ratio'], 'Sustained ratio differs')
    storage = load(evidence / 'campaign-storage-progress.json')
    rollup = load(evidence / 'campaign-storage-rollup.json')
    require(set(storage) == {label for label, sample in samples.items() if not sample['preflight']}, 'Storage coverage differs')
    require(sum(entry['storage_samples'] for entry in storage.values()) == rollup['storage_samples'] == 4157, 'Storage sample sum differs')
    job_path = '/sys/fs/cgroup/system.slice/slurmstepd.scope/job_61375111/'
    require(max(entry['memory_peak_bytes_by_path'][job_path + 'memory.current'] for entry in storage.values()) == rollup['job_memory_peak_bytes'], 'Storage peak differs')
    require(all(not any(entry['memory_events'][job_path + 'memory.events'].values()) for entry in storage.values()), 'Nonzero job memory events')
    for name, expected in rollup['reclaim'].items():
        deltas = [entry['job_memory_stat'][name]['last'] - entry['job_memory_stat'][name]['first'] for entry in storage.values()]
        require((len(deltas), min(deltas), max(deltas), sum(deltas)) == (expected['samples_observed'], expected['minimum_delta'], expected['maximum_delta'], expected['sum_sample_deltas']), 'Reclaim aggregation differs')
    for markdown in (repository / REPORT, repository / SCRIPTS / 'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)', markdown.read_text()):
            if not target.startswith(('http://', 'https://', '#')):
                require((markdown.parent / target.split('#')[0]).exists(), 'Broken Markdown link: ' + target)
    return dict(package_files=len(files) + 1, byte_preserved_files=len(mapping), original_checksum_entries=len(original), preflights=12, timed_samples=48, pairs=24, measured_head=HEAD, manifest_sha256=sha256(evidence / 'manifest.json'), limitations=['Recorded event digests, not discarded event-array replay.', 'Storage rollup checked against derived summaries; raw storage logs unavailable.', 'Sizing binding uses original-tree hash index, not omitted raw files.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[5])
    args = parser.parse_args()
    print(json.dumps(verify(args.repository), indent=2))


if __name__ == '__main__':
    main()

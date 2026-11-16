"""Prepare small hash-frozen snapshots, or print a reviewed sbatch command only.

prepare --attempt ABS --node sdfampere036 --budget JSON --retention JSON
        --partition-evidence JSON --walltime-seconds INT
command --attempt ABS --review JSON

Partition evidence: status=pass, partition=ampere, max_time_seconds=INT,
observed_at=ISO8601, command=TEXT; the parent's availability.json with
partition_max_time and commands is also supported. Neither promises availability.
Review: status=pass, scope=implementation, manifest_sha256=HEX, reviewer=TEXT,
blocking_findings=[], native_compute_compatibility_reviewed=true, evidence=TEXT.
No command is emitted for plan-only review or incomplete parent integration.
"""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import node_selection

from plan import (HERE, REPO, SCRATCH, RECEIPT, RECEIPT_SHA, PR_SHA, CALIBRATION,
                  CONFIG_MISMATCHES, NATIVE_CAVEAT, CONTROLS, RESOURCES, SHARED_RESERVE,
                  DEFAULT_RETENTION, matrix, require, sha, write, source_hashes, historical, slurm_seconds,
                  validate_budget, validate_retention, retained_bytes)


def native_dependencies():
    require(sha(RECEIPT) == RECEIPT_SHA, 'Existing bounce attempt01 receipt changed')
    receipt = json.loads(RECEIPT.read_text())
    require(receipt['status'] == 'pass' and receipt['pr_sha'] == PR_SHA, 'Pinned PR298 required')
    require(receipt['config_mismatches'] == CONFIG_MISMATCHES, 'Unexpected native caveat')
    source, frozen = Path(receipt['source']), Path(receipt['frozen_build']) / 'source'
    require(subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == PR_SHA,
            'Pinned source HEAD changed')
    files = {str(RECEIPT): RECEIPT_SHA}
    files.update({str(source / 'psana/psana' / relative): digest for relative, digest in receipt['python'].items()})
    for relative, digest in receipt['native_sources'].items():
        files[str(source / relative)] = digest
        files[str(frozen / relative)] = digest
    for relative, row in receipt['build_configurations'].items():
        files[str(source / relative)] = row['pr']
        if row['frozen'] is not None:
            files[str(frozen / relative)] = row['frozen']
    for key in ('native', 'kvikio_identity'):
        require(receipt[key], 'Missing runtime inventory: ' + key)
        files.update(receipt[key])
    for path, digest in files.items():
        require(sha(path) == digest, 'Read-only runtime changed: ' + path)
    return receipt, files


def reference_dependencies():
    manifest = CALIBRATION / 'manifest.json'
    files = {str(manifest): sha(manifest)}
    ctypes = set()
    for row in json.loads(manifest.read_text())['calibration']:
        ctype = row['source_document']['ctype']
        if row['detector'] == 'jungfrau' and ctype in ('pedestals', 'pixel_gain', 'pixel_offset'):
            path = CALIBRATION / row['file']
            require(path.resolve().is_relative_to(CALIBRATION), 'Calibration escape')
            require(sha(path) == row['sha256'], 'Calibration payload mismatch')
            files[str(path)] = row['sha256']
            ctypes.add(ctype)
    require(ctypes == {'pedestals', 'pixel_gain', 'pixel_offset'}, 'Incomplete frozen constants')
    return files


def partition_record(record):
    if 'partition_max_time' in record:
        require('scontrol show partition ampere' in record['commands'], 'Missing ampere partition observation')
        return dict(record, status='pass', partition='ampere', command='scontrol show partition ampere',
                    max_time_seconds=slurm_seconds(record['partition_max_time']))
    return record


def prepare(attempt, node, budget, retention, partition, walltime, allowlist=None):
    partition = partition_record(partition)
    require(attempt.is_absolute() and attempt == attempt.resolve() and attempt.is_relative_to(SCRATCH)
            and not attempt.exists(), 'Use a fresh explicit shared-scratch attempt')
    if allowlist is None:
        require(node is not None and re.fullmatch(r'sdfampere\d{3}', node) is not None, 'Explicit inventoried node required')
    else:
        require(node is None, 'Choose one node or a reviewed allowlist, not both')
        node_selection.validate(allowlist)
        live_nodes = subprocess.check_output(['sinfo', '-p', 'ampere', '-N', '-h', '-o', '%N'], text=True, timeout=30).split()
        node_selection.scheduler_flags(allowlist, live_nodes)
    require(SCRATCH.stat().st_uid == os.geteuid() and os.access(SCRATCH, os.W_OK), 'Scratch ownership/access')
    require(shutil.disk_usage(SCRATCH).free >= SHARED_RESERVE, 'Insufficient compact evidence capacity')
    require(partition['status'] == 'pass' and partition['partition'] == 'ampere'
            and partition['command'], 'Read-only partition evidence required')
    observed = datetime.fromisoformat(partition['observed_at'])
    require(observed.tzinfo is not None and 0 <= time.time() - observed.timestamp() <= 86400,
            'Partition evidence must be from the last 24 hours')
    total = validate_budget(budget, walltime, partition['max_time_seconds'])
    validate_retention(retention)
    files = source_hashes()
    spec = importlib.util.spec_from_file_location('compute_campaign_reference', HERE / 'reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    sample_indices = reference.samples().tolist()
    native, external = native_dependencies()
    external.update(reference_dependencies())
    if allowlist is not None:
        for filename, digest in allowlist['evidence_files'].items():
            require(sha(filename) == digest, 'Hardware inventory changed: ' + filename)
        external.update(allowlist['evidence_files'])
    with historical() as helpers:
        fixture, streams, stamps = helpers.support.fixture()
        generation = helpers.support.ROOT / 'generation.json'
        external[str(generation)] = sha(generation)
        timestamp_sha = helpers.support.raw.ts_hash(stamps)
    stage_files = [dict(stream=stream['stream'], kind=kind, **row)
                   for stream in streams for kind, row in stream['files'].items()]
    snapshot_bytes = sum((REPO / relative).stat().st_size for relative in files)
    require(snapshot_bytes <= 128 << 20, 'Snapshot is not a compact source subset')
    require(shutil.disk_usage(SCRATCH).free >= SHARED_RESERVE + snapshot_bytes, 'Snapshot/evidence capacity')
    attempt.mkdir(parents=True, mode=0o700, exist_ok=False)
    for relative, digest in files.items():
        target = attempt / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
        require(sha(target) == digest, 'Snapshot source changed while copying')
        target.chmod(0o444)
    manifest = dict(schema_version=2 if allowlist else 1, files=files, read_only_files=external,
                    native_receipt=str(RECEIPT), native_receipt_sha256=RECEIPT_SHA,
                    pr_sha=PR_SHA, pr_source=native['source'], native_caveat=NATIVE_CAVEAT,
                    repo_head=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                    snapshot_policy='Subset hashes include untracked task code; no clean-commit claim',
                    controls=CONTROLS, resources=RESOURCES, cases=matrix(), node=node,
                    node_availability='unconfirmed; no idle-node or reservation claim',
                    walltime_seconds=walltime, budget_seconds=budget, budget_total_seconds=total,
                    partition_evidence=partition, retention=retention,
                    local_required_bytes=retained_bytes(matrix(), retention),
                    input=dict(generation=str(generation), generation_sha256=external[str(generation)],
                               timestamp_sha256=timestamp_sha, events=len(stamps), files=stage_files,
                               required_bytes=sum(row['bytes'] for row in stage_files)),
                    sampling=dict(seed=20261115, indices=sample_indices,
                                  implementation='source/notes/sim10_compute_cube_20261115/reference.py',
                                  policy='Frozen reference.samples(); every active bin/panel/BD, full empty-bin check'),
                    created_at=time.time(), prepared_not_executed=True)
    if allowlist is not None:
        manifest['node_allowlist'] = allowlist
    write(attempt / 'manifest.json', manifest)
    (attempt / 'manifest.json').chmod(0o444)
    record = dict(status='prepared_not_reviewed_or_submitted', manifest_sha256=sha(attempt / 'manifest.json'),
                  attempt=str(attempt), command=None, native_caveat=NATIVE_CAVEAT,
                  local_required_bytes=manifest['local_required_bytes'],
                  next_gate='Independent review of exact integrated snapshot; new submission authority remains required')
    write(attempt / 'preparation.json', record)
    return record


def verify_manifest(attempt, review_path):
    path = attempt / 'manifest.json'
    review = json.loads(review_path.read_text())
    require(review['status'] == 'pass' and review['scope'] == 'implementation'
            and review['manifest_sha256'] == sha(path) and review['blocking_findings'] == []
            and review['native_compute_compatibility_reviewed'] is True
            and review['reviewer'].strip() and review['evidence'].strip(), 'Valid exact-snapshot implementation review required')
    manifest = json.loads(path.read_text())
    require(manifest['schema_version'] in (1, 2) and manifest['pr_sha'] == PR_SHA
            and manifest['native_receipt_sha256'] == RECEIPT_SHA, 'Manifest runtime/schema changed')
    if manifest['schema_version'] == 2:
        require(manifest['node'] is None, 'Allowlist cannot also pin a node')
        node_selection.validate(manifest['node_allowlist'])
        for name, candidate in manifest['node_allowlist']['candidates'].items():
            require(manifest['read_only_files'].get(candidate['source']) == candidate['sha256'], 'Missing inventory dependency')
            inventory = json.loads(Path(candidate['source']).read_text())
            require(inventory['host'].split('.')[0] == name
                    and node_selection.profile(inventory) == candidate['profile'], 'Inventory profile substitution')
    else:
        require(not manifest.get('node_allowlist'), 'Legacy manifest cannot carry an allowlist')
    require(manifest['cases'] == matrix() and manifest['controls'] == CONTROLS
            and manifest['resources'] == RESOURCES and manifest['native_caveat'] == NATIVE_CAVEAT, 'Campaign contract changed')
    root = attempt / 'source'
    require(root.resolve() == root, 'Snapshot root escape')
    for relative, digest in manifest['files'].items():
        target = root / relative
        require(target.resolve().is_relative_to(root) and not target.is_symlink(), 'Snapshot escape')
        require(sha(target) == digest, 'Frozen source changed: ' + relative)
    require('notes/' + HERE.name + '/plan.py' in manifest['files'], 'Missing frozen campaign plan')
    require(manifest['files'] == source_hashes_at(root), 'Incomplete or changed snapshot subset')
    for filename, digest in manifest['read_only_files'].items():
        require(sha(filename) == digest, 'Read-only dependency changed: ' + filename)
    require(manifest['read_only_files'].get(str(RECEIPT)) == RECEIPT_SHA, 'Missing receipt dependency')
    receipt = json.loads(RECEIPT.read_text())
    require(manifest['pr_source'] == receipt['source'] and manifest['native_receipt'] == str(RECEIPT), 'Runtime source substitution')
    for relative, digest in receipt['python'].items():
        require(manifest['read_only_files'].get(str(Path(receipt['source']) / 'psana/psana' / relative)) == digest,
                'Missing pinned Python dependency')
    for kind in ('native', 'kvikio_identity'):
        require(all(manifest['read_only_files'].get(path) == digest for path, digest in receipt[kind].items()),
                'Missing native/KvikIO dependency')
    validate_budget(manifest['budget_seconds'], manifest['walltime_seconds'],
                    manifest['partition_evidence']['max_time_seconds'])
    validate_retention(manifest['retention'])
    require(manifest['local_required_bytes'] == retained_bytes(matrix(), manifest['retention']), 'Local retention capacity changed')
    return manifest


def source_hashes_at(root):
    spec = importlib.util.spec_from_file_location('sim10_frozen_campaign_plan', root / 'notes' / HERE.name / 'plan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_hashes()


def command(attempt, review):
    manifest = verify_manifest(attempt, review)
    minutes = (manifest['walltime_seconds'] + 59) // 60
    require(minutes * 60 <= manifest['partition_evidence']['max_time_seconds'], 'Rounded Slurm time exceeds partition maximum')
    if manifest.get('node_allowlist'):
        live_nodes = subprocess.check_output(['sinfo', '-p', 'ampere', '-N', '-h', '-o', '%N'], text=True, timeout=30).split()
        selection = node_selection.scheduler_flags(manifest['node_allowlist'], live_nodes)
    else:
        selection = ['--nodelist=' + manifest['node']]
    return ['sbatch', '--parsable', *selection, '--time=' + str(minutes),
            '--chdir=' + str(attempt), '--output=' + str(attempt / 'slurm-%j.out'),
            '--error=' + str(attempt / 'slurm-%j.err'),
            str(attempt / 'source/notes' / HERE.name / 'run.sbatch'), str(attempt), str(review), sha(review)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--attempt', type=Path, required=True)
    target = prep.add_mutually_exclusive_group(required=True)
    target.add_argument('--node')
    target.add_argument('--node-allowlist', type=Path)
    for name in ('budget', 'partition-evidence'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--retention', type=Path)
    prep.add_argument('--walltime-seconds', type=int, required=True)
    display = subs.add_parser('command')
    display.add_argument('--attempt', type=Path, required=True)
    display.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.attempt, args.node, json.loads(args.budget.read_text()),
              json.loads(args.retention.read_text()) if args.retention else DEFAULT_RETENTION,
              json.loads(args.partition_evidence.read_text()),
              args.walltime_seconds, json.loads(args.node_allowlist.read_text()) if args.node_allowlist else None), indent=2))
    else:
        print(shlex.join(command(args.attempt.resolve(), args.review.resolve())))


if __name__ == '__main__':
    main()

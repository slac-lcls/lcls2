"""Snapshot reviewed candidates without commits, worktrees, reservations or job submission."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from plan import HERE, CONTROLS, RESOURCES, BUDGET_SECONDS, matrix, source_hashes
from support import require, sha, write
from prepare_runtime import PR_SHA

SCRATCH = Path('/sdf/scratch/users/m/monarin/gpu-validation')
RECEIPT = SCRATCH / 'sim10-kvikio-bounce-20261113/attempt01/native-compatibility.json'


def prepare(attempt, node):
    attempt = attempt.resolve()
    require(attempt.is_relative_to(SCRATCH) and not attempt.exists(), 'Use a new shared-scratch attempt')
    require(SCRATCH.stat().st_uid == os.geteuid() and os.access(SCRATCH, os.W_OK), 'Shared scratch ownership/access')
    require(shutil.disk_usage(SCRATCH).free >= (2 << 30), 'Insufficient shared-scratch evidence space')
    require(node in ('sdfampere017', 'sdfampere018', 'sdfampere036'), 'Node requires inventory review')
    native = json.loads(RECEIPT.read_text())
    require(native['status'] == 'pass' and native['pr_sha'] == PR_SHA, 'Pinned runtime receipt')
    repo = HERE.parents[1]
    files = source_hashes()
    attempt.mkdir(parents=True, exist_ok=False)
    for filename, digest in files.items():
        relative = Path(filename).relative_to(repo)
        target = attempt / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, target)
        require(sha(target) == digest, 'Snapshot copy changed')
    manifest = dict(files={str(Path(path).relative_to(repo)): digest for path, digest in files.items()},
                    pr_sha=PR_SHA, native_receipt=str(RECEIPT), native_receipt_sha256=sha(RECEIPT),
                    repo_head=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                    snapshot_policy='Explicit file hashes include untracked harness; no clean-commit claim',
                    controls=CONTROLS, resources=RESOURCES, walltime_budget_seconds=BUDGET_SECONDS,
                    cases=matrix(), node=node)
    write(attempt / 'manifest.json', manifest)
    digest = sha(attempt / 'manifest.json')
    command = ['sbatch', '--parsable', '--nodelist=' + node, '--chdir=' + str(attempt),
               '--output=' + str(attempt / 'slurm-%j.out'), '--error=' + str(attempt / 'slurm-%j.err'),
               str(attempt / 'source/notes/sim10_storage_gpu_20261114/run.sbatch'),
               str(attempt), 'REPLACE_AFTER_INDEPENDENT_REVIEW', node]
    record = dict(status='prepared_not_submitted', manifest_sha256=digest, command=command,
                  next_gate='Independent review of this exact snapshot; pass its digest only after approval',
                  live_nvme_qualified=False, submitted=False)
    write(attempt / 'preparation.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--node', choices=('sdfampere017', 'sdfampere018', 'sdfampere036'), default='sdfampere036')
    arguments = parser.parse_args()
    prepare(arguments.attempt, arguments.node)

"""Bounded staging, scheduler deadline guards, and independent cleanup receipts."""
from datetime import datetime
import hashlib
import os
import shutil
import subprocess
import time

from support import require
from storage import owned_stage

CLEANUP_SECONDS = 300


def job_end_time():
    value = os.environ.get('SLURM_JOB_END_TIME')
    if value is None:
        job = os.environ['SLURM_JOB_ID']
        require(job.isdigit(), 'Invalid Slurm job ID')
        value = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%e'],
                                        text=True, timeout=10).strip()
    deadline = float(value) if value.isdigit() else datetime.fromisoformat(value).timestamp()
    require(deadline > time.time(), 'No valid future Slurm end time')
    return deadline


def ensure_time(deadline, operation, ceiling, now=None):
    remaining = deadline - (time.time() if now is None else now)
    if remaining < ceiling + CLEANUP_SECONDS:
        raise RuntimeError('Insufficient walltime for %s: %.1fs left, need %ds including cleanup' %
                           (operation, remaining, ceiling + CLEANUP_SECONDS))
    return remaining


def copy_checked(source, target, expected_bytes, expected_sha, stop, deadline):
    def check():
        if stop.is_set() or time.time() >= deadline:
            raise RuntimeError('Staging interrupted or staging deadline exceeded')

    check()
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        while True:
            check()
            block = incoming.read(8 << 20)
            if not block:
                break
            check()
            outgoing.write(block)
    checksum = hashlib.sha256()
    with target.open('rb') as incoming:
        while True:
            check()
            block = incoming.read(8 << 20)
            if not block:
                break
            checksum.update(block)
    require(target.stat().st_size == expected_bytes and checksum.hexdigest() == expected_sha,
            'Staged identity')


def cleanup_stages(stages, safe):
    receipts = []
    for media, stage in stages.items():
        row = dict(media=media, stage=str(stage), removed=False)
        try:
            if safe:
                shutil.rmtree(owned_stage(stage, media))
                row['removed'] = True
            else:
                row['error'] = 'Live process group not ruled out; preserving stage'
        except Exception as error:
            row['error'] = str(error)
        receipts.append(row)
    return receipts

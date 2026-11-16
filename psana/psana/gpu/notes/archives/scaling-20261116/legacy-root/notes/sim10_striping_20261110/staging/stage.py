"""Create fresh fixed-layout copies and independently hash their full contents."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import time

COUNTS = (1, 2, 4, 8, 16, 32)
CHUNK = 8 * 1024**2
LFS = '/usr/bin/lfs'
CAMPAIGN = Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110')
MANIFEST_SHA256 = '78d78b269125b8f38e5e83b61651344e85b2331c127aca71595266aab7fdface'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(path):
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, f'not private regular file: {path}')
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def layout(path, count):
    raw = command(LFS, 'getstripe', str(path))
    def field(name):
        found = re.findall(r'^' + name + r':\s*(\S+)', raw, re.M)
        require(len(found) == 1, f'not one simple layout: {path}: {name}')
        return found[0]
    require(int(field('lmm_stripe_count')) == count, f'wrong stripe count: {path}')
    require(int(field('lmm_stripe_size')) == 1048576, f'wrong stripe size: {path}')
    require(field('lmm_pool') == 'original', f'wrong pool: {path}')
    require(field('lmm_pattern') == 'raid0', f'wrong layout pattern: {path}')
    osts = [int(x) for x in re.findall(r'^\s+(\d+)\s+\d+\s+0x[0-9a-f]+\s+0x[0-9a-f]+\s*$', raw, re.M)]
    require(len(osts) == count and len(set(osts)) == count, f'wrong OST list: {path}')
    return {'count': count, 'size': 1048576, 'pool': 'original', 'osts': osts, 'raw': raw}


def digest(path, expected):
    before = identity(path)
    h = hashlib.sha256()
    total = 0
    with path.open('rb', buffering=0) as stream:
        while True:
            data = stream.read(CHUNK)
            if not data:
                break
            total += len(data)
            h.update(data)
    require(identity(path) == before, f'file changed while hashing: {path}')
    require(total == expected['bytes'], f'wrong byte count: {path}')
    require(h.hexdigest() == expected['sha256'], f'SHA256 mismatch: {path}')
    return {'path': str(path), 'bytes': total, 'sha256': h.hexdigest(), 'identity': before}


def copy_one(source, dest, expected, count):
    initial = identity(source)
    require(initial['st_size'] == expected['bytes'], f'wrong source size: {source}')
    require(not dest.exists() and not dest.is_symlink(), f'destination exists: {dest}')
    command(LFS, 'setstripe', '-c', str(count), '-S', '1M', '-p', 'original', str(dest))
    created = identity(dest)
    require(created['st_size'] == 0, f'destination not empty: {dest}')
    before_layout = layout(dest, count)
    started = time.monotonic()
    total = 0
    # No truncation, replacement, reflinks, sparse skipping or copy_file_range.
    fd = os.open(dest, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino, opened.st_size) ==
                (created['st_dev'], created['st_ino'], 0), 'destination changed before write')
        with source.open('rb', buffering=0) as stream:
            while True:
                data = stream.read(CHUNK)
                if not data:
                    break
                view = memoryview(data)
                while view:
                    n = os.write(fd, view)
                    require(n > 0, f'zero write: {dest}')
                    total += n
                    view = view[n:]
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    after = identity(dest)
    require(identity(source) == initial, f'source changed during copy: {source}')
    require(total == expected['bytes'] == after['st_size'], f'incomplete copy: {dest}')
    require((created['st_dev'], created['st_ino']) == (after['st_dev'], after['st_ino']), 'inode replaced')
    after_layout = layout(dest, count)
    require(before_layout == after_layout, 'layout changed during copy')
    return {'path': str(dest), 'bytes': total, 'copy_seconds': time.monotonic() - started,
            'identity': after, 'layout': after_layout}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'large I/O requires a Slurm allocation')
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    require(CAMPAIGN.resolve() in output.parents, 'output must be within the campaign root')
    require(not output.exists(), 'output directory must be new; existing data will not be overwritten')
    require(source != output and source not in output.parents and output not in source.parents,
            'source and output must be disjoint')
    manifest_bytes = args.manifest.read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256,
            'manifest differs from accepted transfer manifest')
    rows = [r for r in json.loads(manifest_bytes)['files'] if r['kind'] == 'bd']
    names = [f'mfx101555026-r0010-s00{i}-c000.xtc2' for i in range(5, 10)]
    require(len(rows) == 5 and sorted(r['relative_path'] for r in rows) == names, 'wrong source manifest')
    for r in rows:
        require(type(r['bytes']) is int and r['bytes'] > 0 and
                re.fullmatch('[0-9a-f]{64}', r['sha256']), 'bad size/hash')
    rows.sort(key=lambda r: r['relative_path'])
    require(os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize >
            sum(r['bytes'] for r in rows) * len(COUNTS) * 1.1, 'insufficient free space')
    output.mkdir(mode=0o700)
    started = time.monotonic()
    status = {'state': 'source_verification', 'started_unix': time.time(),
              'job': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
              'source': str(source), 'counts': list(COUNTS), 'workers': 5,
              'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'verification_mode': 'separate full buffered read after fsync/close and advisory DONTNEED; cache residency and remote bytes unverified',
              'sets': []}
    def update():
        status['updated_unix'] = time.time()
        status['elapsed_seconds'] = time.monotonic() - started
        save(output / 'status.json', status)
        print(json.dumps({k: status[k] for k in ('state', 'updated_unix', 'elapsed_seconds')},
                         sort_keys=True), flush=True)
    def terminate(signum, frame):
        status.update(state='failed', error=f'terminated by signal {signum}')
        update()
        raise RuntimeError(status['error'])
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        update()
        with ThreadPoolExecutor(max_workers=5) as pool:
            sources = list(pool.map(lambda r: digest(source / r['relative_path'], r), rows))
            status['sources'] = sources
            source_done = time.monotonic()
            status['source_verify_seconds'] = source_done - started
            for count in COUNTS:
                directory = output / f'count-{count:02d}'
                directory.mkdir()
                status.update(state='copying', current_count=count)
                update()
                begin = time.monotonic()
                copies = list(pool.map(lambda r: copy_one(source / r['relative_path'],
                                       directory / r['relative_path'], r, count), rows))
                copied = time.monotonic()
                status['state'] = 'verifying'
                update()
                verified = list(pool.map(lambda r: digest(directory / r['relative_path'], r), rows))
                for receipt, copied_receipt in zip(verified, copies):
                    require(receipt['identity'] == copied_receipt['identity'], 'copy changed before verification')
                    require(layout(Path(receipt['path']), count) == copied_receipt['layout'], 'layout changed')
                record = {'count': count, 'copies': copies, 'verified': verified,
                          'copy_wall_seconds': copied - begin,
                          'verify_wall_seconds': time.monotonic() - copied,
                          'state': 'verified'}
                save(directory / 'verified.json', record)
                status['sets'].append(record)
                elapsed_sets = time.monotonic() - source_done
                status['estimated_remaining_seconds'] = elapsed_sets / len(status['sets']) * (6-len(status['sets']))
                status['state'] = 'set_verified'
                update()
        for receipt in status['sources']:
            require(identity(Path(receipt['path'])) == receipt['identity'], 'source changed during campaign')
        for group in status['sets']:
            for receipt in group['verified']:
                path = Path(receipt['path'])
                require(identity(path) == receipt['identity'], 'verified output changed later')
                layout(path, group['count'])
        require(len(status['sets']) == 6, 'missing sets')
        status.update(state='complete', completed_unix=time.time(), estimated_remaining_seconds=0)
        update()
        save(output / 'complete.json', status)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        update()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    main()

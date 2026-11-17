"""Independent run-51 SMD reference and compute-node private-copy preparation.

scan reads only small SMD files; prepare copies/hashes on an allocated node.
No DataSource/Events/benchmark iteration, GPU initialization, or cache eviction.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from native_contract import digest, file_hash, input_ranges, read, require, write

STREAMS = [3, 5, 7, 8, 9]
EVENTS = 10000
AVAILABLE = 17872
BLOCK = 16 << 20


def stamp(path):
    st = Path(path).stat()
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns, device=st.st_dev, inode=st.st_ino)


def header(raw):
    require(len(raw) == 24, 'short XTC header')
    low, high, env, src, damage_type, extent = struct.unpack('<6I', raw)
    require(extent >= 12, 'invalid XTC extent')
    return (high << 32) | low, (env >> 24) & 15, extent + 12


def merge_ranges(rows):
    ranges = []
    for timestamp, step, offset, size, service in rows:
        require(offset >= 0 and size >= 24, 'invalid SMD offset/size')
        if ranges and ranges[-1][0] + ranges[-1][1] == offset:
            ranges[-1][1] += size
        else:
            require(not ranges or offset > sum(ranges[-1]), 'overlapping SMD offsets')
            ranges.append([offset, size])
    return ranges


def scan_stream(smd, bigdata, Dgram):
    """Decode only SMD; explicit framing prevents truncation being treated as EOF."""
    before = stamp(smd)
    require(before['size'] <= 32 << 20, 'SMD exceeds bounded metadata scan')
    raw = smd.read_bytes()
    require(stamp(smd) == before and len(raw) == before['size'], 'SMD changed during scan')
    offset, config, step, begun = 0, None, -1, False
    rows, transitions, segments = [], [], []
    while offset < len(raw):
        timestamp, service, length = header(raw[offset:offset+24])
        require(offset + length <= len(raw), 'truncated SMD record')
        view = memoryview(raw)[offset:offset+length]
        if config is None:
            require(service == 2, 'SMD must start with Configure')
            config = Dgram(view=view)
            segments = sorted(int(s) for s in config.software.jungfrau)
        else:
            require(service != 2, 'multiple Configure records unsupported')
            d = Dgram(config=config, view=view)
            require(int(d.timestamp()) == timestamp and d.service() == service, 'decoder/header disagreement')
            if service == 4:
                info = d.runinfo[0].runinfo
                require(not begun and info.expt == 'mfx100848724' and info.runnum == 51, 'wrong BeginRun')
                begun = True
            if service == 6:
                step += 1
            if service in (11, 12):
                require(begun and step >= 0, 'event outside run/step')
                a = d.smdinfo[0].offsetAlg
                row = [timestamp, step, int(a.intOffset), int(a.intDgramSize), service]
                require(not rows or timestamp > rows[-1][0], 'unordered/duplicate SMD event')
                rows.append(row)
            else:
                transitions.append([timestamp, service, len(rows)])
        offset += length
    require(len(rows) == AVAILABLE, 'unexpected available event count')
    ranges = merge_ranges(rows[:EVENTS])
    big = dict(path=str(bigdata), **stamp(bigdata), ranges=ranges,
               range_bytes=sum(length for start, length in ranges))
    input_ranges(big)
    return dict(stream=int(smd.name.split('-s')[1][:3]), segments=segments,
                bigdata=big, smd=dict(path=str(smd), sha256=hashlib.sha256(raw).hexdigest(), **before),
                events=rows, transitions=transitions)


def source_paths(root):
    expected = {'bigdata': [], 'smd': []}
    for stream in STREAMS:
        stem = 'mfx100848724-r0051-s%03d-c000' % stream
        expected['bigdata'].append(root / (stem+'.xtc2'))
        expected['smd'].append(root / 'smalldata' / (stem+'.smd.xtc2'))
    # Reject extra chunks for the selected streams rather than silently ignoring them.
    for key, folder, suffix in (('bigdata', root, '.xtc2'), ('smd', root/'smalldata', '.smd.xtc2')):
        for stream, path in zip(STREAMS, expected[key]):
            matches = sorted(folder.glob('mfx100848724-r0051-s%03d-c*%s' % (stream, suffix)))
            require(matches == [path] and path.is_file() and not path.is_symlink(), 'unexpected source chunks/links')
    return expected


def scan(root, output):
    import numpy  # initialize decoder prerequisites through the qualified CPU install
    from psana import dgram
    root, output = root.resolve(strict=True), output.absolute()
    require(not output.exists(), 'scan output already exists')
    paths = source_paths(root)
    streams = [scan_stream(smd, big, dgram.Dgram) for smd, big in zip(paths['smd'], paths['bigdata'])]
    identities = [[r[0], r[1]] for r in streams[0]['events']]
    transitions = streams[0]['transitions']
    require(all([[r[0], r[1]] for r in s['events']] == identities and s['transitions'] == transitions
                for s in streams), 'streams disagree on event/step/transition identities')
    require(sorted(seg for s in streams for seg in s['segments']) == list(range(32)), 'wrong/duplicate JF segments')
    events = [[row[0], row[1], sum(s['events'][i][3] for s in streams)] for i, row in enumerate(identities[:EVENTS])]
    reference = dict(schema='run51-raw-reference-v1', events=events)
    decoder = Path(dgram.__file__).resolve()
    plan = dict(schema='run51-input-plan-v1', source=str(root), streams=streams, reference=reference,
                event_sha256=digest([e[0] for e in events]), requested_bytes=sum(e[2] for e in events),
                available_events=AVAILABLE, selected_events=EVENTS,
                python=sys.executable, python_version=sys.version,
                decoder=dict(path=str(decoder), sha256=file_hash(decoder)),
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(output, plan)
    print(json.dumps(dict(plan=str(output), sha256=file_hash(output), events=len(events),
                         copy_bytes=sum(s[k]['size'] for s in streams for k in ('bigdata','smd')),
                         requested_bytes=plan['requested_bytes'], ranges_per_stream=[len(s['bigdata']['ranges']) for s in streams])), flush=True)


def verify_plan(plan):
    require(plan['schema'] == 'run51-input-plan-v1' and [s['stream'] for s in plan['streams']] == STREAMS, 'wrong plan')
    require(plan['selected_events'] == EVENTS and plan['available_events'] == AVAILABLE, 'wrong prefix')
    reference = plan['reference']
    require(reference['schema'] == 'run51-raw-reference-v1' and len(reference['events']) == EVENTS, 'wrong reference')
    expected = []
    for i in range(EVENTS):
        rows = [s['events'][i] for s in plan['streams']]
        require(all(r[:2] == rows[0][:2] and r[4] in (11,12) for r in rows), 'event identity differs')
        expected.append(rows[0][:2] + [sum(r[3] for r in rows)])
    require(reference['events'] == expected and plan['event_sha256'] == digest([e[0] for e in expected]) and
            plan['requested_bytes'] == sum(e[2] for e in expected), 'reference differs from SMD plan')
    for s in plan['streams']:
        require(s['bigdata']['ranges'] == merge_ranges(s['events'][:EVENTS]), 'ranges differ from SMD')
        input_ranges(s['bigdata'])
        for key in ('bigdata','smd'):
            entry = s[key]
            require(stamp(entry['path']) == {k:entry[k] for k in ('size','mtime_ns','device','inode')}, 'source changed')
        require(file_hash(s['smd']['path']) == s['smd']['sha256'], 'source SMD changed')
    return reference


def hash_contents(path, ranges):
    full, selected, index, position = hashlib.sha256(), hashlib.sha256(), 0, 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(BLOCK), b''):
            full.update(block)
            end = position + len(block)
            while index < len(ranges) and ranges[index][0] < end:
                start, length = ranges[index]
                stop = start + length
                selected.update(block[max(0,start-position):min(len(block),stop-position)])
                if stop <= end:
                    index += 1
                else:
                    break
            position = end
    require(index == len(ranges), 'hash range exceeds file')
    return full.hexdigest(), selected.hexdigest()


def copy_checked(entry, target, rows=None):
    source = Path(entry['path'])
    initial = stamp(source)
    require(initial == {k:entry[k] for k in initial} and not source.is_symlink(), 'source changed before copy')
    source_hash = hashlib.sha256()
    with source.open('rb') as src, target.open('xb') as dst:
        require(stamp(source) == initial, 'source replaced during open')
        for block in iter(lambda: src.read(BLOCK), b''):
            dst.write(block)
            source_hash.update(block)
        dst.flush()
        os.fsync(dst.fileno())
    require(stamp(source) == initial and target.stat().st_size == initial['size'], 'source changed during copy')
    ranges = input_ranges(entry) if rows is not None else []
    full_hash, selected_hash = hash_contents(target, ranges)
    require(full_hash == source_hash.hexdigest(), 'copy hash mismatch')
    if rows is None:
        require(full_hash == entry['sha256'], 'copied SMD differs from reference scan')
    else:
        with target.open('rb', buffering=0) as stream:
            # Sparse header checks must not trigger large sequential read-ahead.
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            for timestamp, step, offset, size, service in rows:
                stream.seek(offset)
                require(header(stream.read(24)) == (timestamp, service, size), 'SMD/bigdata header mismatch')
            # This fixed run has only 24-byte SlowUpdate records between L1 ranges.
            for left, right in zip(ranges, ranges[1:]):
                gap_start, gap_end = sum(left), right[0]
                require(gap_end-gap_start == 24, 'unqualified inter-event gap')
                stream.seek(gap_start)
                _, service, length = header(stream.read(24))
                require(service == 10 and length == 24, 'gap is not a SlowUpdate')
    target.chmod(0o444)
    require(target.stat().st_uid == os.getuid() and target.stat().st_nlink == 1, 'nonprivate copy')
    result = dict(path=str(target), sha256=selected_hash if rows is not None else full_hash, **stamp(target))
    if rows is not None:
        result.update(ranges=ranges, range_bytes=entry['range_bytes'])
    receipt = dict(source=str(source), target=str(target), source_stat=initial,
                   source_sha256=source_hash.hexdigest(), copy_sha256=full_hash, verified=True)
    print(json.dumps(dict(copied=str(target), bytes=initial['size'], sha256=full_hash)), flush=True)
    return result, receipt


def prepare(plan_path, plan_sha, destination):
    require(os.environ.get('SLURM_JOB_ID') and os.uname().nodename.startswith('nid'), 'compute allocation required')
    require(file_hash(plan_path) == plan_sha, 'plan hash differs')
    plan = read(plan_path)
    reference = verify_plan(plan)
    destination = destination.absolute()
    require(destination.parent.resolve() == destination.parent and not destination.exists(), 'new nonsymlink destination required')
    copy_bytes = sum(s[k]['size'] for s in plan['streams'] for k in ('bigdata','smd'))
    require(shutil.disk_usage(destination.parent).free > copy_bytes + (10 << 30), 'insufficient free storage')
    destination.mkdir(mode=0o700)
    write(destination/'preparation-started.json', dict(plan=str(plan_path), plan_sha256=plan_sha,
          job=os.environ['SLURM_JOB_ID'], node=os.uname().nodename, copy_bytes=copy_bytes))
    # Explicit reproducible layout, independent of parent directory defaults.
    subprocess.run(['lfs','setstripe','-c','1','-S','1048576',str(destination)], check=True)
    (destination/'smalldata').mkdir(mode=0o700)
    def stream_copy(s):
        big, big_receipt = copy_checked(s['bigdata'], destination/Path(s['bigdata']['path']).name, s['events'][:EVENTS])
        smd, smd_receipt = copy_checked(s['smd'], destination/'smalldata'/Path(s['smd']['path']).name)
        layout = subprocess.check_output(['lfs','getstripe',big['path']], text=True)
        count = subprocess.check_output(['lfs','getstripe','-c',big['path']], text=True).strip()
        size = subprocess.check_output(['lfs','getstripe','-S',big['path']], text=True).strip()
        require(count == '1' and size == '1048576', 'destination stripe layout differs')
        return dict(stream=s['stream'], bigdata=big, smd=smd), [big_receipt, smd_receipt], layout
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(stream_copy, plan['streams']))
    verify_plan(plan)  # source metadata/SMD unchanged throughout all copies
    write(destination/'event-reference.json', reference)
    manifest = dict(schema='run51-private-input-v2', private_copy=True, root=str(destination),
                    event_reference_sha256=file_hash(destination/'event-reference.json'), files=[r[0] for r in results])
    write(destination/'dataset-manifest.json', manifest)
    write(destination/'preparation-complete.json', dict(complete=True, job=os.environ['SLURM_JOB_ID'],
          node=os.uname().nodename, plan_sha256=plan_sha, generator_sha256=file_hash(__file__),
          helper_sha256=file_hash(Path(__file__).with_name('native_contract.py')),
          manifest_sha256=file_hash(destination/'dataset-manifest.json'),
          event_reference_sha256=manifest['event_reference_sha256'], event_sha256=plan['event_sha256'],
          requested_bytes=plan['requested_bytes'], events=EVENTS, available_events=AVAILABLE,
          source_copy_receipts=[receipt for result in results for receipt in result[1]],
          stripe_layouts=[result[2] for result in results],
          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          scope='Input preparation only; no runtime, cache-state or performance qualification.'))
    print('INPUT PREPARATION COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scan_args = sub.add_parser('scan')
    scan_args.add_argument('--source', type=Path, required=True)
    scan_args.add_argument('--output', type=Path, required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--plan', type=Path, required=True)
    prep.add_argument('--plan-sha256', required=True)
    prep.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'scan':
        scan(args.source, args.output)
    else:
        prepare(args.plan, args.plan_sha256, args.destination)


if __name__ == '__main__':
    main()

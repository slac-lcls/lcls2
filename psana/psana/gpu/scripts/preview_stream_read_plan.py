"""Print proposed per-stream requests from real SMD metadata; no GPU/I/O timing.

This preview groups the aligned benchmark's real SMD rows into batch_size
event batches. It does not run EventBuilder or issue the planned BigData reads.
Full plans go to JSON on scratch; only the first few batches are printed.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys


def read_stream(directory, name, count):
    from psana import dgram
    smd = directory/'smalldata'/name.replace('.xtc2', '.smd.xtc2')
    file_size = (directory/name).stat().st_size
    fd = os.open(smd, os.O_RDONLY)
    rows, transitions = [], []
    try:
        config = dgram.Dgram(file_descriptor=fd)
        while len(rows) < count:
            event = dgram.Dgram(config=config)
            service, timestamp = event.service(), int(event.timestamp())
            if service != 12:
                transitions.append((timestamp, service))
                continue
            locator = event.smdinfo[0].offsetAlg
            size, offset = int(locator.intDgramSize), int(locator.intOffset)
            if size < 0 or offset < 0 or offset+size > file_size:
                raise ValueError(f'invalid SMD extent in {smd}')
            rows.append(dict(timestamp=timestamp, offset=offset, size=size,
                             fence=len(transitions)))
    finally:
        os.close(fd)
    return dict(smd=str(smd), rows=rows, transitions=transitions,
                descriptors_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-manifest', type=Path,
                   help='Cross-check event count, timestamp hashes and bytes against this baseline')
    p.add_argument('--planner-source', type=Path,
                   help='Load this new planner against an existing native psana installation')
    p.add_argument('--events', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--print-batches', type=int, default=3)
    p.add_argument('--small-target-bytes', type=int, default=1 << 20)
    p.add_argument('--input-capacity-mib', type=int, default=64,
                   help='Raw-input-only progress allowance, not the total GPU memory budget')
    a = p.parse_args()
    if min(a.events, a.batch_size, a.print_batches) <= 0:
        p.error('events, batch-size and print-batches must be positive')
    if a.planner_source:
        spec = importlib.util.spec_from_file_location('psana.gpu.gpu_stream_read_plan', a.planner_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    from psana.gpu import gpu_stream_read_plan as planner
    from psana.gpu.gpu_read_plan import ResolvedDgram, ResolvedFile
    streams = (0, 5, 6, 7, 8, 9)
    directory = a.directory.resolve()
    names = {s: f'mfx101210926-r0387-s{s:03d}-c000.xtc2' for s in streams}
    inputs = {s: read_stream(directory, name, a.events) for s, name in names.items()}
    reference = inputs[0]
    for s in streams:
        assert inputs[s]['transitions'] == reference['transitions'], 'fixture transition mismatch'
        assert [(r['timestamp'],r['fence']) for r in inputs[s]['rows']] == [
            (r['timestamp'],r['fence']) for r in reference['rows']], 'fixture is not event-aligned'
    if a.reference_manifest:
        manifest = json.loads(a.reference_manifest.read_text())
        assert manifest['events'] == a.events
        expected = {r['name']: r for r in manifest['streams']}
        assert set(expected) == set(names.values())
        for s, name in names.items():
            rows = inputs[s]['rows']
            assert sum(r['size'] for r in rows) == expected[name]['payload_bytes']
            stamp_hash = hashlib.sha256(struct.pack(f'<{len(rows)}Q',
                *(r['timestamp'] for r in rows))).hexdigest()
            assert stamp_hash == expected[name]['timestamp_sha256']
    plans = []
    print(f'SMD PLAN ONLY: {a.events} events, batch_size={a.batch_size}, '
          f'target={a.small_target_bytes} B; printing first {a.print_batches} batches')
    print('Each request gets independent backing; offsets below are FILE offsets.')
    for batch, base in enumerate(range(0, a.events, a.batch_size)):
        stop = min(base+a.batch_size, a.events)
        rows = [ResolvedDgram(i-base, inputs[s]['rows'][i]['timestamp'], s,
                             ResolvedFile(str(directory/names[s]), 0),
                             inputs[s]['rows'][i]['offset'], inputs[s]['rows'][i]['size'])
                for i in range(base, stop) for s in streams]
        fences = {i-base: reference['rows'][i]['fence'] for i in range(base, stop)}
        plan = planner.build_stream_read_plan(rows, batch_id=batch, n_events=stop-base,
            small_target_bytes=a.small_target_bytes, input_capacity_bytes=a.input_capacity_mib*2**20,
            fence_by_event=fences)
        plans.append(dict(event_base=base, plan=asdict(plan)))
        if batch >= a.print_batches:
            continue
        print(f'\nBatch {batch}: events [{base},{stop}), requests={len(plan.groups)}, '
              f'one-group/stream input bound={plan.one_group_per_stream_bytes:,} B')
        print('| Stream | Requests | Event coverage | Request sizes B | First request: events; offset; bytes |')
        print('|---|---:|---|---|---|')
        for s in streams:
            groups = [g for g in plan.groups if g.stream_id == s]
            g = groups[0]
            sizes = sorted({g.size for g in groups})
            print(f'| s{s:03d} | {len(groups)} | [{base+g.first_event},{base+groups[-1].event_stop}) | '
                  f'{sizes} | [{base+g.first_event},{base+g.event_stop}); {g.file_offset}; {g.size} |')
        for s in streams:
            small = [g for g in plan.groups if g.stream_id == s and g.small]
            if small:
                print(f's{s:03d} small reads: ' + '; '.join(
                    f'g{g.group_id} events[{base+g.first_event},{base+g.event_stop}) '
                    f'offset={g.file_offset} bytes={g.size} after_group={g.after_group}'
                    for g in small[:6]))
                if len(small) > 6:
                    print(f'  ... {len(small)-6} further small groups in JSON')
        offered = ', '.join(f's{g.stream_id:03d}[{base+g.first_event},{base+g.event_stop})'
                            for g in plan.groups[:11])
        print('First submissions: '+offered)
    total = sum(x['plan']['useful_bytes'] for x in plans)
    output = dict(dataset='mfx101210926/r0387', events=a.events, batch_size=a.batch_size,
                  directory=str(directory), useful_bytes=total, plans=plans,
                  planner=str(Path(planner.__file__).resolve()),
                  planner_sha256=hashlib.sha256(Path(planner.__file__).read_bytes()).hexdigest(),
                  reference_manifest=str(a.reference_manifest) if a.reference_manifest else None,
                  inputs=inputs, mode='SMD-only proposal; no BigData requests issued')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(f'\nAll {len(plans)} batches: {sum(len(x["plan"]["groups"]) for x in plans)} requests; '
          f'{total:,} exact input bytes. Full per-stream plans: {a.output}')


if __name__ == '__main__':
    main()

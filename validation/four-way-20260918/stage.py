"""Stage bounded, unmodified XTC prefixes for the acceptance benchmark."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct


def inspect_file(path, count):
    offset = payload = 0
    timestamps = []
    with open(path, "rb", buffering=0) as stream:
        while len(timestamps) < count:
            stream.seek(offset)
            header = stream.read(24)
            if len(header) != 24:
                raise RuntimeError(f"short header {path}:{offset}")
            timestamp, env = struct.unpack_from("<QI", header)
            extent = struct.unpack_from("<I", header, 20)[0]
            if extent < 12:
                raise RuntimeError(f"invalid extent {path}:{offset}")
            size = 12 + extent
            if (env >> 24) & 15 == 12:
                timestamps.append(timestamp)
                payload += size
            offset += size
    if offset > path.stat().st_size:
        raise RuntimeError(f"truncated last dgram {path}")
    return dict(name=path.name, events=count, payload_bytes=payload,
                last_end=offset, stage_bytes=min(path.stat().st_size, offset + 16 * 1024**2),
                timestamp_sha256=hashlib.sha256(struct.pack(f"<{count}Q", *timestamps)).hexdigest())


def inspect_smd(path, count):
    from psana import dgram
    smd = path.parent / 'smalldata' / path.name.replace('.xtc2', '.smd.xtc2')
    fd = os.open(smd, os.O_RDONLY)
    timestamps = []
    payload = end = 0
    try:
        config = dgram.Dgram(file_descriptor=fd)
        while len(timestamps) < count:
            event = dgram.Dgram(config=config)
            if event.service() != 12:
                continue
            locator = event.smdinfo[0].offsetAlg
            size, offset = int(locator.intDgramSize), int(locator.intOffset)
            payload += size
            end = max(end, offset + size)
            timestamps.append(int(event.timestamp()))
    finally:
        os.close(fd)
    if end > path.stat().st_size:
        raise RuntimeError(f'SMD references past source end: {path}')
    return dict(name=path.name, events=count, payload_bytes=payload, last_end=end,
                stage_bytes=min(path.stat().st_size, end + 16 * 1024**2),
                timestamp_sha256=hashlib.sha256(struct.pack(f'<{count}Q', *timestamps)).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--streams", type=int, nargs="+", required=True)
    p.add_argument("--events", type=int, required=True)
    a = p.parse_args()
    target = Path(a.target)
    target.mkdir(parents=True, exist_ok=False)
    (target / "smalldata").mkdir()
    paths = [Path(a.source) / f"{a.exp}-r{a.run:04d}-s{s:03d}-c000.xtc2" for s in a.streams]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(paths)) as pool:
        records = list(pool.map(lambda path: inspect_smd(path, a.events), paths))
    assert len({r["timestamp_sha256"] for r in records}) == 1, "selected streams are not equal-rate/aligned"
    required = sum(r["stage_bytes"] for r in records)
    assert shutil.disk_usage(target).free > required + 20 * 1024**3
    print('STAGE_PLAN ' + json.dumps(records), flush=True)

    def copy(item):
        source, record = item
        remain = record["stage_bytes"]
        with open(source, "rb", buffering=0) as src, open(target / source.name, "xb", buffering=0) as dst:
            while remain:
                data = src.read(min(remain, 16 * 1024**2))
                if not data:
                    raise RuntimeError(f"short source {source}")
                view = memoryview(data)
                while view:
                    written = dst.write(view)
                    if not written:
                        raise RuntimeError("short stage write")
                    view = view[written:]
                remain -= len(data)
            os.fsync(dst.fileno())
        smd = source.name.replace(".xtc2", ".smd.xtc2")
        shutil.copyfile(Path(a.source) / "smalldata" / smd, target / "smalldata" / smd)
        local = inspect_file(target / source.name, a.events)
        for key in ('timestamp_sha256', 'payload_bytes', 'last_end'):
            assert local[key] == record[key], (source, key, local[key], record[key])
        print("STAGED " + json.dumps(record), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(paths)) as pool:
        list(pool.map(copy, zip(paths, records)))
    manifest = dict(exp=a.exp, run=a.run, events=a.events, streams=records,
                    payload_bytes=sum(r["payload_bytes"] for r in records), stage_bytes=required,
                    timestamp_sha256=records[0]["timestamp_sha256"])
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("STAGE_MANIFEST " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

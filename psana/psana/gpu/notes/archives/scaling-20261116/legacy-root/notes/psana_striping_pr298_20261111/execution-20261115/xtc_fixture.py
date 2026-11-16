"""Small, checked XTC2 helpers for the frozen MFX run-10 pilot (not general DAQ IO)."""
import hashlib
import json
import math
import struct
from pathlib import Path

HEADER = struct.Struct("<QIIHHI")
XTC = struct.Struct("<IHHI")
COUNTS = [19, 20, 21]
SOURCE_EXP = "mfx101555026"
DEST_EXP = "prjdatpsana"
SNAPSHOT = Path("/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage1-20261103")
DEST = Path("/sdf/data/lcls/ds/prj/prjdatpsana/xtc")
OLD_NOTES = Path(__file__).resolve().parent.parent / "synthetic_mfx_stage1_20261103"
SIZES = [1, 2, 4, 8, 1, 2, 4, 8, 4, 8, 1, 4, 4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def header(data):
    require(len(data) >= 24, "Short datagram")
    h = HEADER.unpack_from(data)
    require(h[-1] >= 12 and h[-1] + 12 == len(data), "Datagram extent mismatch")
    return h


def service(data):
    return (header(data)[1] >> 24) & 15


def records(blob):
    pos = 0
    while pos < len(blob):
        require(pos + 24 <= len(blob), "Truncated header")
        size = struct.unpack_from("<I", blob, pos + 20)[0] + 12
        require(size >= 24 and pos + size <= len(blob), "Truncated payload")
        yield pos, blob[pos:pos + size]
        pos += size


def read_record(f, offset):
    f.seek(offset)
    raw = f.read(24)
    require(len(raw) == 24, "Short source header")
    extent = HEADER.unpack(raw)[-1]
    require(12 <= extent <= 128 * 1024 * 1024, "Unexpected datagram size")
    data = raw + f.read(extent - 12)
    header(data)
    return data


def children(data, start=12, end=None):
    """Yield XTC headers recursively; Data/Names contents are opaque here."""
    if end is None:
        end = len(data)
    require(start + 12 <= end, "Short XTC")
    src, damage, contains, extent = XTC.unpack_from(data, start)
    require(extent >= 12 and start + extent == end, "XTC extent mismatch")
    yield start, src, contains & 4095, extent
    if contains & 4095 in (0, 1):  # Parent or ShapesData
        pos = start + 12
        while pos < end:
            require(pos + 12 <= end, "Short child XTC")
            size = XTC.unpack_from(data, pos)[-1]
            require(size >= 12 and pos + size <= end, "Bad child extent")
            yield from children(data, pos, pos + size)
            pos += size


def cstring(data):
    return data.split(b"\0", 1)[0].decode("utf-8")


def schemas(config):
    """Read stable Names wire layout from ShapesData.hh, validating every size."""
    out = {}
    for pos, src, kind, size in children(config):
        if kind != 4:
            continue
        require(size >= 1048 and (size - 1048) % 524 == 0, "Unexpected Names layout")
        require(src not in out and src < 1 << 20, "Duplicate/invalid NamesId")
        fields = []
        for p in range(pos + 1048, pos + size, 524):
            name = cstring(config[p + 260:p + 516])
            dtype, rank = struct.unpack_from("<II", config, p + 516)
            require(dtype < len(SIZES) and rank < 5, "Unsupported field layout")
            fields.append((name, dtype, rank))
        num_arrays = struct.unpack_from("<I", config, pos + 12)[0]
        require(num_arrays == sum(rank > 0 for _, _, rank in fields), "Array count mismatch")
        out[src] = {"detname": cstring(config[pos + 272:pos + 528]),
                    "dettype": cstring(config[pos + 16:pos + 272]),
                    "detid": cstring(config[pos + 528:pos + 784]),
                    "alg": cstring(config[pos + 784:pos + 1040]),
                    "version": struct.unpack_from("<I", config, pos + 1040)[0],
                    "segment": struct.unpack_from("<I", config, pos + 1044)[0],
                    "fields": fields, "arrays": num_arrays}
    return out


def schema_for(all_schemas, name):
    matches = [(key, value) for key, value in all_schemas.items() if value["detname"] == name]
    require(len(matches) == 1, "Expected one schema for " + name)
    return matches[0]


def spans(data, names_id, schema):
    matches = [(p, size) for p, src, kind, size in children(data) if kind == 1 and src == names_id]
    require(len(matches) == 1, "Expected one ShapesData for NamesId")
    pos, size = matches[0]
    parts = list(children(data, pos, pos + size))[1:]
    shapes = [(p + 12, n - 12) for p, _, kind, n in parts if kind == 2]
    payloads = [(p + 12, n - 12) for p, _, kind, n in parts if kind == 3]
    require(len(payloads) == 1, "Expected one Data child")
    require(len(shapes) == 1 and shapes[0][1] == schema["arrays"] * 20, "Unexpected Shapes layout")
    p, length = payloads[0]
    end, array_index, out = p + length, 0, {}
    for name, dtype, rank in schema["fields"]:
        count = 1
        if rank:
            dimensions = struct.unpack_from("<5I", data, shapes[0][0] + array_index * 20)
            count = math.prod(dimensions[:rank])
            array_index += 1
        size = count * SIZES[dtype]
        require(name not in out and p + size <= end, "Invalid data field span")
        out[name] = (p, size, dtype, rank)
        p += size
    require(p == end, "Unaccounted Data bytes")
    return out


def smd_offsets(data, config_schemas):
    key, schema = schema_for(config_schemas, "smdinfo")
    fields = spans(data, key, schema)
    require(set(fields) == {"intOffset", "intDgramSize"}, "Unexpected smdinfo")
    result = []
    for name in ("intOffset", "intDgramSize"):
        pos, size, dtype, rank = fields[name]
        require((size, dtype, rank) == (8, 3, 0), "Non-UINT64 offset")
        result.append((pos, struct.unpack_from("<Q", data, pos)[0]))
    return result


def to_ns(timestamp):
    seconds, ns = timestamp >> 32, timestamp & 0xffffffff
    require(ns < 1_000_000_000, "Unnormalized timestamp")
    return seconds * 1_000_000_000 + ns


def from_ns(ns):
    seconds, remainder = divmod(ns, 1_000_000_000)
    require(0 <= seconds < 1 << 32, "Timestamp overflow")
    return (seconds << 32) | remainder


def insert_times(left, right, count):
    a, b = to_ns(left), to_ns(right)
    delta = min(1000, (b - a - 1) // count)
    require(delta >= 1, "No space for inserted transitions")
    return [from_ns(a + j * delta) for j in range(1, count + 1)]


def check_stat(path, expected):
    s = path.stat()
    require((s.st_size, s.st_mtime_ns, s.st_ino) ==
            (expected["size"], expected["mtime_ns"], expected["inode"]), "Source file identity changed: " + str(path))


def load_source(entry, snapshot):
    """Index SMD plus sparse bigdata headers; never read all detector data."""
    path = Path(entry["xtc"]["path"])
    check_stat(path, entry["xtc"])
    smd = (snapshot / "smd" / Path(entry["smd"]["path"]).name).read_bytes()
    require(sha(smd) == entry["smd"]["sha256"], "Frozen SMD hash changed")
    frames = list(records(smd))
    cfg = frames[0][1]
    sc = schemas(cfg)
    rows, cursor = [], 0
    with path.open("rb") as f:
        bd_config = read_record(f, 0)
        require(sha(bd_config) == entry["configure_sha256"], "Bigdata Configure changed")
        for offset, raw in frames:
            h = header(raw)
            svc = (h[1] >> 24) & 15
            if svc == 12:
                (_, source_offset), (_, bd_size) = smd_offsets(raw, sc)
                require(source_offset == cursor, "SMD offset discontinuity")
            elif svc == 10:
                # Smd::generate copies SlowUpdate verbatim. The next L1 offset
                # independently checks this inferred length; retained records
                # are also read/compared in full by generation and validation.
                bd_size = len(raw)
            else:
                f.seek(cursor)
                bh = f.read(24)
                require(len(bh) == 24, "Missing bigdata transition")
                parsed = HEADER.unpack(bh)
                require(parsed[:5] == h[:5], "Transition header mismatch")
                bd_size = parsed[-1] + 12
                require(bd_size >= 24, "Bad bigdata extent")
            rows.append({"smd_offset": offset, "smd_size": len(raw),
                         "bd_offset": cursor, "bd_size": bd_size,
                         "timestamp": h[0], "service": svc})
            cursor += bd_size
    require(cursor == entry["xtc"]["size"], "Unindexed bigdata tail")
    require(sum(r["service"] == 12 for r in rows) == 36037, "Source event count changed")
    require(all(a["timestamp"] < b["timestamp"] for a, b in zip(rows, rows[1:])), "Unordered source")
    check_stat(path, entry["xtc"])
    return {"entry": entry, "path": path, "smd": smd, "smd_config": cfg,
            "bd_config": bd_config, "schemas": sc, "rows": rows}


def load_sources(snapshot=SNAPSHOT):
    raw = (snapshot / "manifest.json").read_bytes()
    require(raw == (OLD_NOTES / "manifest.json").read_bytes(), "Snapshot manifest differs from checkpoint")
    manifest = json.loads(raw)
    sources = []
    for index, entry in enumerate(manifest["streams"]):
        sources.append(load_source(entry, snapshot))
        print(f"Indexed frozen SMD and source transition headers: stream {index}", flush=True)
    require(len(sources) == 10, "Expected ten streams")
    reference = [(r["timestamp"], r["service"]) for r in sources[0]["rows"]]
    for source in sources:
        require([(r["timestamp"], r["service"]) for r in source["rows"]] == reference, "Unaligned source streams")
    return sources


def schedule(rows):
    out, event_count, step = [], 0, 0
    for index, row in enumerate(rows):
        svc = row["service"]
        out.append({"source_index": index, "timestamp": row["timestamp"],
                    "service": svc, "step": step, "inserted": False})
        if svc != 12:
            continue
        event_count += 1
        if event_count not in (19, 39, 60):
            continue
        services = [9, 7, 5] if event_count == 60 else [9, 7, 6, 8]
        times = insert_times(row["timestamp"], rows[index + 1]["timestamp"], len(services))
        for new_service, timestamp in zip(services, times):
            if new_service == 6:
                step += 1
            template = next(i for i, r in enumerate(rows) if r["service"] == new_service)
            out.append({"source_index": template, "timestamp": timestamp,
                        "service": new_service, "step": step, "inserted": True})
        if event_count == 60:
            break
    require(event_count == 60 and len(out) == 76, "Unexpected pilot schedule")
    require(all(a["timestamp"] < b["timestamp"] for a, b in zip(out, out[1:])), "Unordered output schedule")
    audit = json.loads((OLD_NOTES / "timestamp_schedule_audit.json").read_text())["pilot"]
    expected = [pair for boundary in audit["boundaries"] for pair in boundary["inserted"]]
    require([[r["timestamp"], r["service"]] for r in out if r["inserted"]] == expected, "Schedule differs from approved audit")
    return out

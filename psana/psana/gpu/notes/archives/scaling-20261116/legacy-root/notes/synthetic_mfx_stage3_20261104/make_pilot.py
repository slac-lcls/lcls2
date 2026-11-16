"""Generate only the approved 60-event run-1 pilot into a new staging directory."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import numpy as np
from psana import dgram, dgramedit
from psana.dgramedit import AlgDef, DetectorDef, DgramEdit

from xtc_fixture import (COUNTS, DEST_EXP, HEADER, SNAPSHOT, check_stat, children,
                         file_sha, header, load_sources, read_record, require,
                         save_json, schedule, schema_for, schemas, sha,
                         smd_offsets, spans)
import struct
import json


def append_payload(source, payload):
    require(header(source)[4] & 4095 == 0, "Expected Parent XTC")
    out = bytearray(source + payload)
    struct.pack_into("<I", out, 20, len(out) - 12)
    list(children(out))
    return out


def scan_blocks(names_id):
    cfg = DgramEdit(transition_id=2, ts=1, bufsize=65536)
    det = cfg.Detector(DetectorDef("scan", "scan", "1234"), AlgDef("raw", 1, 0, 0),
                       {"step_value": (np.int64, 0), "step_docstring": (str, 1)},
                       nodeId=names_id >> 8, namesId=names_id & 255, segment=0)
    buf = bytearray(65536)
    cfg.save(buf)
    config = bytes(buf[:cfg.size])
    require(set(schemas(config)) == {names_id}, "Unexpected generated scan Names")
    steps, start = [], 0
    for index, count in enumerate(COUNTS):
        step = DgramEdit(transition_id=6, ts=2, config_dgramedit=cfg, bufsize=65536)
        det.raw.step_value = index
        det.raw.step_docstring = json.dumps({"detname": "scan", "scantype": "synthetic_mfx",
                                            "step": index, "source_exp": "mfx101555026",
                                            "source_run": 10, "source_event_range": [start, start + count]})
        step.adddata(det.raw)
        step.save(buf)
        raw = bytes(buf[:step.size])
        decoded = dgram.Dgram(config=dgram.Dgram(view=config), view=raw)
        require(decoded.scan[0].raw.step_value == index, "Scan creation failed")
        steps.append(raw[24:])
        start += count
    return config[24:], steps


def rewrite_runinfo(data, config_schemas):
    key, schema = schema_for(config_schemas, "runinfo")
    fields = spans(data, key, schema)
    require(set(fields) == {"expt", "runnum"}, "Unexpected runinfo fields")
    p, size, dtype, rank = fields["expt"]
    require((dtype, rank) == (10, 1) and size > len(DEST_EXP.encode()), "Experiment field too small or wrong type")
    require(data[p:p + size].split(b"\0", 1)[0] == b"mfx101555026", "Wrong source experiment")
    data[p:p + size] = DEST_EXP.encode().ljust(size, b"\0")
    p, size, dtype, rank = fields["runnum"]
    require((size, dtype, rank) == (4, 2, 0), "Expected UINT32 run number")
    require(struct.unpack_from("<I", data, p)[0] == 10, "Wrong source run")
    struct.pack_into("<I", data, p, 1)


def transform(raw, item, stream, config_schemas, scan_names, scan_steps, bd_offset=None, bd_size=None):
    out = bytearray(raw)
    struct.pack_into("<Q", out, 0, item["timestamp"])
    svc = item["service"]
    require((header(raw)[1] >> 24) & 15 == svc, "Wrong template service")
    if svc == 2 and stream == 0:
        out = append_payload(out, scan_names)
    elif svc == 4:
        rewrite_runinfo(out, config_schemas)
    elif svc == 6 and stream == 0:
        out = append_payload(out, scan_steps[item["step"]])
    elif svc == 12 and bd_offset is not None:
        (offset_pos, _), (size_pos, _) = smd_offsets(out, config_schemas)
        struct.pack_into("<Q", out, offset_pos, bd_offset)
        struct.pack_into("<Q", out, size_pos, bd_size)
    header(out)
    return out


def make(staging, sources):
    staging.mkdir(mode=0o700, parents=False, exist_ok=False)
    (staging / "smalldata").mkdir()
    used = set(schemas(sources[0]["bd_config"])) | set(sources[0]["schemas"])
    names_id = next((n for n in reversed(range(256)) if n not in used), None)
    require(names_id is not None, "No free node-0 NamesId")
    scan_names, scan_steps = scan_blocks(names_id)
    plan = schedule(sources[0]["rows"])
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "destination": {"experiment": DEST_EXP, "run": 1},
                "staging": str(staging.resolve()), "step_counts": COUNTS,
                "scan_names_id": names_id, "schedule": plan, "streams": [],
                "runtime": {"python": sys.version, "executable": sys.executable,
                            "dgram": dgram.__file__, "dgramedit": dgramedit.__file__},
                "code_sha256": {p.name: file_sha(p) for p in
                    [Path(__file__), Path(__file__).with_name("xtc_fixture.py")]}}
    for stream, source in enumerate(sources):
        names = {"bd": f"{DEST_EXP}-r0001-s{stream:03d}-c000.xtc2",
                 "smd": f"smalldata/{DEST_EXP}-r0001-s{stream:03d}-c000.smd.xtc2"}
        trace, bd_config, smd_config = [], None, None
        with source["path"].open("rb") as original, (staging / names["bd"]).open("xb") as bd, \
                (staging / names["smd"]).open("xb") as smd:
            for item in plan:
                row = source["rows"][item["source_index"]]
                raw_bd = read_record(original, row["bd_offset"])
                require(len(raw_bd) == row["bd_size"], "Source bigdata size changed")
                raw_smd = source["smd"][row["smd_offset"]:row["smd_offset"] + row["smd_size"]]
                require(header(raw_bd)[:2] == header(raw_smd)[:2], "Source event header mismatch")
                bd_pos, smd_pos = bd.tell(), smd.tell()
                new_bd = transform(raw_bd, item, stream, schemas(source["bd_config"]) if item["service"] == 4 else {},
                                   scan_names, scan_steps)
                new_smd = transform(raw_smd, item, stream, source["schemas"], scan_names, scan_steps,
                                    bd_offset=bd_pos, bd_size=len(new_bd))
                if item["service"] == 2:
                    bd_config = dgram.Dgram(view=bytes(new_bd))
                    smd_config = dgram.Dgram(view=bytes(new_smd))
                else:
                    for data, cfg in ((new_bd, bd_config), (new_smd, smd_config)):
                        decoded = dgram.Dgram(config=cfg, view=bytes(data))
                        if item["service"] == 4:
                            require(decoded.runinfo[0].runinfo.expt == DEST_EXP and
                                    decoded.runinfo[0].runinfo.runnum == 1, "Runinfo decode failed")
                        if item["service"] == 6 and stream == 0:
                            require(decoded.scan[0].raw.step_value == item["step"], "BeginStep decode failed")
                bd.write(new_bd)
                smd.write(new_smd)
                trace.append({"source_index": item["source_index"],
                              "source_bd_offset": row["bd_offset"], "source_smd_offset": row["smd_offset"],
                              "source_bd_sha256": sha(raw_bd), "source_smd_sha256": sha(raw_smd),
                              "bd_offset": bd_pos, "bd_size": len(new_bd),
                              "smd_offset": smd_pos, "smd_size": len(new_smd)})
            for f in (bd, smd):
                f.flush()
                os.fsync(f.fileno())
        check_stat(source["path"], source["entry"]["xtc"])
        manifest["streams"].append({"stream": stream, "source": source["entry"],
                                    "files": {kind: {"relative_path": name, "size": (staging / name).stat().st_size,
                                                      "sha256": file_sha(staging / name)} for kind, name in names.items()},
                                    "records": trace})
        print(f"Wrote and hashed stream {stream}: 76 records, 60 events", flush=True)
    save_json(staging / "generation.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    require(os.environ.get("PS_FAKESTEP_FLAG", "0") == "0", "Disable fake steps")
    make(args.staging, load_sources(args.snapshot))

"""Read every pilot record and independently enforce allowed source-byte changes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct

from psana import dgram
from xtc_fixture import (COUNTS, DEST_EXP, SNAPSHOT, check_stat, children, file_sha,
                         header, load_sources, read_record, require, save_json,
                         schedule, schema_for, schemas, sha, smd_offsets, spans)


def compare_allowed(original, output, svc, stream, original_schemas, output_schemas, scan_id):
    """Restore only allowed changed bytes, then require full source equality."""
    restored = bytearray(output)
    # Only inserted transitions can change timestamp; caller checks exact times.
    restored[:8] = original[:8]
    if svc in (2, 6) and stream == 0:
        require(len(output) > len(original), "Missing appended scan block")
        root_children = list(children(output))
        kind = 4 if svc == 2 else 1
        added = [(p, n) for p, src, k, n in root_children if p >= len(original) and k == kind and src == scan_id]
        require(added == [(len(original), len(output) - len(original))], "Unexpected appended scan data")
        restored = restored[:len(original)]
        restored[20:24] = original[20:24]
    elif svc == 4:
        key, schema = schema_for(original_schemas, "runinfo")
        for name, (p, size, _, _) in spans(original, key, schema).items():
            require(name in ("expt", "runnum"), "Unexpected runinfo field")
            restored[p:p + size] = original[p:p + size]
    elif svc == 12 and any(v["detname"] == "smdinfo" for v in output_schemas.values()):
        # Only SMD L1 carries smdinfo; original small-detector data stays intact.
        for p, _ in smd_offsets(original, original_schemas):
            restored[p:p + 8] = original[p:p + 8]
    require(restored == original, f"Unapproved source-byte change, stream {stream}, service {svc}")


def verify(root, sources):
    generation = json.loads((root / "generation.json").read_text())
    plan = schedule(sources[0]["rows"])
    require(generation["schedule"] == plan and generation["step_counts"] == COUNTS, "Generation schedule mismatch")
    require(generation["destination"] == {"experiment": DEST_EXP, "run": 1}, "Wrong destination")
    require(len(generation["streams"]) == 10, "Missing output stream")
    scan_id = generation["scan_names_id"]
    report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "root": str(root.resolve()),
              "generation_sha256": file_sha(root / "generation.json"),
              "code_sha256": {p.name: file_sha(p) for p in
                  [Path(__file__), Path(__file__).with_name("xtc_fixture.py")]},
              "scope": "Every record, source bytes, schemas, offsets and transition state; not calibrated/MPI acceptance.",
              "streams": []}
    for stream, (source, saved) in enumerate(zip(sources, generation["streams"])):
        require(saved["stream"] == stream and saved["source"] == source["entry"], "Wrong source mapping")
        expected = {"bd": f"{DEST_EXP}-r0001-s{stream:03d}-c000.xtc2",
                    "smd": f"smalldata/{DEST_EXP}-r0001-s{stream:03d}-c000.smd.xtc2"}
        for kind, name in expected.items():
            item = saved["files"][kind]
            require(item["relative_path"] == name, "Noncanonical filename")
            require((root / name).stat().st_size == item["size"] and file_sha(root / name) == item["sha256"], "Output file hash/size changed")
        counts, current, state, event_times = [], -1, "start", []
        configs, out_schemas, original_schemas = {}, {}, {"bd": schemas(source["bd_config"]), "smd": source["schemas"]}
        require(len(saved["records"]) == len(plan), "Missing record mapping")
        cursors = {"bd": 0, "smd": 0}
        with (root / expected["bd"]).open("rb") as bd, (root / expected["smd"]).open("rb") as smd, source["path"].open("rb") as src:
            for index, (item, trace) in enumerate(zip(plan, saved["records"])):
                svc, ts = item["service"], item["timestamp"]
                row = source["rows"][item["source_index"]]
                require(trace["source_index"] == item["source_index"] and
                        trace["source_bd_offset"] == row["bd_offset"] and
                        trace["source_smd_offset"] == row["smd_offset"], "Wrong trace offsets")
                originals = {"bd": read_record(src, row["bd_offset"]),
                             "smd": source["smd"][row["smd_offset"]:row["smd_offset"] + row["smd_size"]]}
                data = {"bd": read_record(bd, cursors["bd"]), "smd": read_record(smd, cursors["smd"])}
                for kind in ("bd", "smd"):
                    raw, old = data[kind], originals[kind]
                    require(header(old)[0] == row["timestamp"], "Source record timestamp mismatch")
                    require(header(raw)[0] == ts and (header(raw)[1] >> 24) & 15 == svc, "Output timeline mismatch")
                    require(header(raw)[1:5] == header(old)[1:5], "env/src/damage/type changed")
                    require(trace[kind + "_offset"] == cursors[kind] and trace[kind + "_size"] == len(raw), "Wrong output mapping")
                    require(trace["source_" + kind + "_sha256"] == sha(old), "Source hash changed")
                    list(children(raw))  # Recursively validate all nested extents.
                    if svc == 2:
                        configs[kind] = dgram.Dgram(view=raw)
                        out_schemas[kind] = schemas(raw)
                        original = original_schemas[kind]
                        require(all(out_schemas[kind].get(k) == v for k, v in original.items()), "Original Names changed")
                        require(set(out_schemas[kind]) - set(original) == ({scan_id} if stream == 0 else set()), "Unexpected Names change")
                        if stream == 0:
                            require(out_schemas[kind][scan_id] == {"detname": "scan", "dettype": "scan", "detid": "1234",
                                    "alg": "raw", "version": 65536, "segment": 0,
                                    "fields": [("step_value", 7, 0), ("step_docstring", 10, 1)], "arrays": 1}, "Wrong scan schema")
                    else:
                        decoded = dgram.Dgram(config=configs[kind], view=raw)
                        if svc == 4:
                            require(decoded.runinfo[0].runinfo.expt == DEST_EXP and decoded.runinfo[0].runinfo.runnum == 1, "Wrong decoded runinfo")
                            pos, size, _, _ = spans(raw, *schema_for(out_schemas[kind], "runinfo"))["expt"]
                            require(raw[pos:pos + size] == b"prjdatpsana".ljust(size, b"\0") and size > len(b"prjdatpsana"), "Runinfo padding changed")
                        if svc == 6:
                            require(hasattr(decoded, "scan") == (stream == 0), "Scan on wrong stream")
                            if stream == 0:
                                scan = decoded.scan[0].raw
                                doc = json.loads(scan.step_docstring)
                                require(scan.step_value == item["step"] == doc["step"], "Wrong scan value")
                                require(doc == {"detname": "scan", "scantype": "synthetic_mfx", "step": item["step"],
                                                "source_exp": "mfx101555026", "source_run": 10,
                                                "source_event_range": [sum(COUNTS[:item["step"]]), sum(COUNTS[:item["step"] + 1])]}, "Wrong scan provenance")
                        if svc == 8:
                            require(not hasattr(decoded, "chunkinfo"), "Unexpected chunk payload")
                        if svc == 12 and kind == "smd":
                            offsets = decoded.smdinfo[0].offsetAlg
                            require(offsets.intOffset == cursors["bd"] and offsets.intDgramSize == len(data["bd"]), "Decoded SMD offset/size mismatch")
                    compare_allowed(old, raw, svc, stream, original_schemas[kind], out_schemas[kind], scan_id)
                if svc == 2:
                    require(state == "start", "Configure state")
                    state = "configured"
                elif svc == 4:
                    require(state == "configured", "BeginRun state")
                    state = "run"
                elif svc == 6:
                    require(state == "run", "BeginStep state")
                    counts.append(0)
                    current += 1
                    require(item["step"] == current, "Step identity")
                    state = "step"
                elif svc == 8:
                    require(state == "step", "Enable state")
                    state = "enabled"
                elif svc in (10, 12):
                    require(state == "enabled", "Data outside enabled step")
                    if svc == 12:
                        counts[-1] += 1
                        event_times.append(ts)
                elif svc == 9:
                    require(state == "enabled", "Disable state")
                    state = "disabled"
                elif svc == 7:
                    require(state == "disabled", "EndStep state")
                    state = "run"
                elif svc == 5:
                    require(state == "run" and index == len(plan) - 1, "EndRun state")
                    state = "done"
                else:
                    raise ValueError("Unexpected service")
                for kind in cursors:
                    cursors[kind] += len(data[kind])
            require(not bd.read(1) and not smd.read(1), "Unexpected output tail")
        require(counts == COUNTS and state == "done", "Incomplete step walk")
        expected_times = [r["timestamp"] for r in source["rows"] if r["service"] == 12][:60]
        require(event_times == expected_times, "Event omission/duplication/timestamp change")
        check_stat(source["path"], source["entry"]["xtc"])
        report["streams"].append({"stream": stream, "records": len(plan), "event_count": 60,
                                  "step_counts": counts, "checks": "pass", "files": saved["files"]})
        print(f"Verified stream {stream}: all source bytes, decoded metadata and SMD offsets", flush=True)
    report["status"] = "pass"
    report["total_bytes"] = sum(f["size"] for s in report["streams"] for f in s["files"].values())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    require(os.environ.get("PS_FAKESTEP_FLAG", "0") == "0", "Disable fake steps")
    save_json(args.report, verify(args.root, load_sources(args.snapshot)))

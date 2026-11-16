"""Freeze source metadata and calibration payloads; never write to a service."""
import argparse
import ast
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import struct
import tempfile
from datetime import datetime, timezone

import numpy as np
import requests

logging.basicConfig(level=logging.WARNING)
original_request = requests.sessions.Session.request


def read_only_request(self, method, url, *args, **kwargs):
    if method.upper() != "GET":
        raise RuntimeError(f"Stage 1 prohibits HTTP {method}")
    kwargs.setdefault("timeout", 120)
    return original_request(self, method, url, *args, **kwargs)


requests.sessions.Session.request = read_only_request

from psana import DataSource
from psana.pscalib.calib import MDBWebUtils as wu
from psana.pscalib.calib import CalibConstants as cc

PUBLIC_CALIB = "https://pswww.slac.stanford.edu/ws/calib_ws/"
assert cc.URL == PUBLIC_CALIB, cc.URL


def public_get(url, query=None, timeout=120, **kwargs):
    # Some installed clients attach Kerberos to public reads. Restrict this
    # transport adapter to the public read endpoint; selection stays unchanged.
    if not url.startswith(PUBLIC_CALIB):
        raise RuntimeError(f"Unexpected calibration read endpoint: {url}")
    response = requests.get(url, params=query, timeout=timeout)
    response.raise_for_status()
    return response


wu.get = public_get
wu.USE_QUERY_STR = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_record(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    output = args.output
    previous_payloads = {}
    if args.resume_incomplete:
        assert (output / "manifest.inprogress.json").is_file()
        assert not (output / "manifest.json").exists()
        previous = json.loads((output / "manifest.inprogress.json").read_text())
        previous_payloads = {r["file"]: r for r in previous["calibration"]}
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "calibration").mkdir()
        (output / "smd").mkdir()
    source = Path("/sdf/data/lcls/ds/prj/public01/xtc")
    dest = Path("/sdf/data/lcls/ds/prj/prjdatpsana")
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "source": {"experiment": "mfx101555026", "run": 10},
                "destination": {"experiment": "prjdatpsana", "run": 1},
                "pilot_step_counts": [19, 20, 21], "streams": [],
                "detectors": {}, "calibration": [], "access": {}}
    for sub in ["xtc", "xtc/smalldata", "results"]:
        path = dest / sub
        collisions = [str(p) for p in path.glob("*r0001-*")] if sub != "results" else []
        if collisions:
            raise RuntimeError(f"Run 1 collision: {collisions}")
        # Unique, non-XTC probe; remove even on failure.
        fd, probe = tempfile.mkstemp(prefix=".stage1-access-", dir=path)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(b"synthetic-mfx-stage1\n")
                stream.flush()
                os.fsync(stream.fileno())
            assert Path(probe).read_bytes() == b"synthetic-mfx-stage1\n"
        finally:
            Path(probe).unlink()
        fs = os.statvfs(path)
        manifest["access"][sub] = {"create_read_fsync_remove": True,
            "run1_collisions": collisions,
            "filesystem_total_bytes": fs.f_blocks * fs.f_frsize,
            "filesystem_available_bytes": fs.f_bavail * fs.f_frsize}
    logbook = "https://pswww.slac.stanford.edu/ws/lgbk/lgbk/prjdatpsana/ws/"
    for name, suffix in [("run1_files", "1/current_files_for_live_mode"), ("runs", "runs")]:
        response = requests.get(logbook + suffix)
        manifest["access"][name] = {"status": response.status_code}
        if response.ok:
            manifest["access"][name]["response"] = response.json()
    listing = manifest["access"]["run1_files"].get("response", {})
    assert listing.get("success") and listing.get("value") == [], listing
    manifest["access"]["calibration_destination_collections"] = public_get(
        PUBLIC_CALIB + "cdb_prjdatpsana").json()
    from krtc import KerberosTicket
    headers = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    for label, url in [
        ("authenticated_runs", "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/prjdatpsana/ws/runs"),
        ("calibration_write_privilege", "https://pswww.slac.stanford.edu/ws-kerb/calib_ws/cdb_prjdatpsana/test_post_privilege"),
    ]:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        manifest["access"][label] = response.json()
    assert manifest["access"]["authenticated_runs"] == {"success": True, "value": []}
    assert manifest["access"]["calibration_write_privilege"].get("status") is True
    save_json(output / "manifest.inprogress.json", manifest)
    print("Destination probes passed; no run-1 files found", flush=True)

    ds = DataSource(exp="mfx101555026", run=10, dir=str(source),
                    skip_calib_load="all", max_events=1)
    run = next(ds.runs())
    assert run.expt == "mfx101555026" and run.runnum == 10
    manifest["source"]["beginrun_timestamp"] = int(run.timestamp)
    all_times = []
    for smd, xtc in zip(ds.smd_files, ds.xtc_files):
        smd, xtc = Path(smd), Path(xtc)
        blob = smd.read_bytes()
        (output / "smd" / smd.name).write_bytes(blob)
        pos, counts, times, headers = 0, {}, [], []
        while pos < len(blob):
            timestamp, env, _, _, extent = struct.unpack_from("<QIIII", blob, pos)
            size = 12 + extent
            assert extent >= 12 and pos + size <= len(blob)
            service = (env >> 24) & 15
            counts[str(service)] = counts.get(str(service), 0) + 1
            if service in (11, 12):
                times.append(timestamp)
            if len(times) <= 61:
                headers.append({"timestamp": timestamp, "service": service,
                                "smd_offset": pos, "smd_size": size})
            pos += size
        assert len(times) == 36037 and len(set(times)) == len(times)
        assert times == sorted(times)
        all_times.append(times)
        with xtc.open("rb") as stream:
            header = stream.read(24)
            extent = struct.unpack_from("<I", header, 20)[0]
            configure = header + stream.read(extent - 12)
        manifest["streams"].append({"xtc": file_record(xtc),
            "smd": dict(file_record(smd), sha256=hashlib.sha256(blob).hexdigest()),
            "configure_sha256": hashlib.sha256(configure).hexdigest(),
            "configure_bytes": len(configure), "services": counts,
            "event_count": len(times), "pilot_and_next_headers": headers})
    assert len(all_times) == 10 and all(t == all_times[0] for t in all_times)
    manifest["source"]["pilot_timestamps"] = all_times[0][:60]
    manifest["source"]["next_event_timestamp"] = all_times[0][60]
    manifest["source"]["full_xtc_hashes"] = "Not computed; size/mtime/inode and Configure hashes recorded. SMD copies fully hashed."
    print("All ten streams contain the same 36,037 event timestamps", flush=True)

    active_detector = None

    def snapshot_data(dbname, doc, **kwargs):
        # Called by the actual psana selection implementation for every selected
        # type, including detector fallback. Stream bytes to disk, not to RAM.
        dtype = doc.get("data_type")
        if dtype not in ("ndarray", "str"):
            raise RuntimeError(f"Unsupported snapshot encoding: {dtype}")
        rel = Path("calibration") / f"{doc['shortname']}-{doc['ctype']}-{doc['_id']}.bin"
        path = output / rel
        url = PUBLIC_CALIB + dbname + "/gridfs/" + doc["id_data"]
        known = {r["file"]: r for r in manifest["calibration"]}
        known.update(previous_payloads)
        if path.exists():
            assert str(rel) in known, f"Untracked existing payload: {path}"
            assert sha(path) == known[str(rel)]["sha256"], f"Snapshot hash mismatch: {path}"
            assert known[str(rel)]["source_db"] == dbname
            assert known[str(rel)]["source_document"]["id_data"] == doc["id_data"]
        else:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with path.open("xb") as stream:
                    for block in response.iter_content(4 * 1024 * 1024):
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
        record = {"detector": active_detector, "source_db": dbname,
                  "source_document": dict(doc), "file": str(rel),
                  "bytes": path.stat().st_size, "sha256": sha(path)}
        if dtype == "ndarray":
            shape = ast.literal_eval(doc["data_shape"])
            dt = np.dtype(doc["data_dtype"])
            assert record["bytes"] == int(np.prod(shape)) * dt.itemsize
            array = np.memmap(path, dtype=dt, shape=shape, mode="r")
            if doc["ctype"] == "pixel_status":
                if active_detector == "jungfrau":
                    record["nonzero_per_gain"] = [int(np.count_nonzero(g)) for g in array]
                    record["pixels_per_gain"] = int(np.prod(shape[1:]))
                    record["nonfinite_per_gain"] = [int(np.count_nonzero(~np.isfinite(g))) for g in array]
                else:
                    record["nonzero_pixels"] = int(np.count_nonzero(array))
                    record["nonfinite_pixels"] = int(np.count_nonzero(~np.isfinite(array)))
            del array
        else:
            path.read_text(encoding="utf-8")
        manifest["calibration"].append(record)
        save_json(output / "manifest.inprogress.json", manifest)
        print(f"Snapshot {active_detector}/{doc['ctype']}: {record['bytes']} bytes", flush=True)
        return record

    wu.get_data_for_doc = snapshot_data
    for name, info in sorted(ds.dsparms.configinfo_dict.items()):
        active_detector = name
        manifest["detectors"][name] = {
            "uniqueid": info.uniqueid, "dettype": info.dettype,
            "segment_ids": info.sorted_segment_ids, "segment_detector_ids": info.detid_dict,
            "stream_indices": [i for i, cfg in enumerate(ds._configs) if hasattr(cfg.software, name)]}
        shortname = wu.pro_detector_name(info.uniqueid, add_shortname=False)
        manifest["detectors"][name]["calibration_shortname"] = shortname
        resolved = wu.calib_constants_all_types(info.uniqueid, exp=run.expt, run=run.runnum) if shortname else {}
        if shortname is None:
            manifest["detectors"][name]["calibration_note"] = "No registered detector-name mapping; no calibration records selected"
        manifest["detectors"][name]["calibration_types"] = sorted((resolved or {}).keys())
        save_json(output / "manifest.inprogress.json", manifest)
        print(f"Resolved {name}: {manifest['detectors'][name]['calibration_types']}", flush=True)
    expected = {"pedestals", "pixel_status", "pixel_rms", "pixel_min", "pixel_max",
                "geometry", "pixel_gain", "pixel_offset"}
    assert set(manifest["detectors"]["jungfrau"]["calibration_types"]) == expected
    assert manifest["detectors"]["jungfrau"]["segment_ids"] == list(range(32))
    manifest["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
        "calibration_module": wu.__file__, "calibration_module_sha256": sha(wu.__file__),
        "selection_source": inspect.getsource(wu.calib_constants_all_types),
        "read_transport": "Public HTTP GET adapter; psana selection unchanged; non-GET requests prohibited"}
    # Verify persisted artifacts again; selection records refer to these exact bytes.
    for record in manifest["calibration"]:
        assert sha(output / record["file"]) == record["sha256"]
    for record in manifest["streams"]:
        path = Path(record["xtc"]["path"])
        assert file_record(path) == record["xtc"], "Source changed during preparation"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    unique_payloads = {x["file"]: x for x in manifest["calibration"]}
    manifest["calibration_total_bytes"] = sum(x["bytes"] for x in unique_payloads.values())
    manifest["unique_calibration_payloads"] = len(unique_payloads)
    manifest["stage1_status"] = "complete; run 1 unused at inspection, not reserved"
    save_json(output / "manifest.json", manifest)
    (output / "manifest.inprogress.json").unlink()
    print(f"Completed snapshot: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Deploy the frozen run-1 snapshot to the experiment DB and verify readback."""
import argparse
import ast
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import requests
from krtc import KerberosTicket
from psana.pscalib.calib import MDBWebUtils as wu
from psana.pscalib.calib import CalibConstants as cc

logging.basicConfig(level=logging.WARNING)
RO = "https://pswww.slac.stanford.edu/ws/calib_ws/"
RW = "https://pswww.slac.stanford.edu/ws-kerb/calib_ws/"
DB = "cdb_prjdatpsana"
COLLECTIONS = {"jungfrau_000009", "pv_000001"}
MANIFEST_SHA = "90a27eed3679111aa24574c09b5505be6b8775429e5b8de10dc0c9de117b983d"
assert cc.URL == RO and cc.URL_KRB == RW
original_request = requests.sessions.Session.request
write_enabled = False
post_log = []
output = None


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def guarded_request(self, method, url, *args, **kwargs):
    method = method.upper()
    allowed = {RW + DB + "/gridfs/"} | {RW + DB + "/" + c + "/" for c in COLLECTIONS}
    if method != "GET" and not (write_enabled and method == "POST" and url in allowed):
        raise RuntimeError(f"Disallowed operation: {method} {url}")
    kwargs.setdefault("timeout", 180)
    response = original_request(self, method, url, *args, **kwargs)
    if method == "POST":
        entry = {"url": url, "status": response.status_code,
                 "utc": datetime.now(timezone.utc).isoformat()}
        try:
            entry["id"] = response.json().get("_id")
        except ValueError:
            entry["non_json_response"] = True
        post_log.append(entry)
        save(output / "post_journal.json", post_log)
    response.raise_for_status()
    return response


requests.sessions.Session.request = guarded_request


def public_get(url, query=None, timeout=180, **kwargs):
    assert url.startswith(RO)
    return requests.get(url, params=query, timeout=timeout)


wu.get = public_get
wu.USE_QUERY_STR = True


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(path, doc):
    if doc["data_type"] == "str":
        return path.read_text(encoding="utf-8")
    assert doc["data_type"] == "ndarray"
    return np.memmap(path, mode="r", dtype=np.dtype(doc["data_dtype"]),
                     shape=ast.literal_eval(doc["data_shape"]))


def payload_sha(data):
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str)
                          else memoryview(np.ascontiguousarray(data)).cast("B")).hexdigest()


def docs(collection):
    return public_get(RO + DB + "/" + collection).json()


def main():
    global output, write_enabled
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()
    output = args.output
    snapshot = args.snapshot
    assert sha(snapshot / "manifest.json") == MANIFEST_SHA
    manifest = json.loads((snapshot / "manifest.json").read_text())
    unique = {}
    for record in manifest["calibration"]:
        doc = record["source_document"]
        key = doc["shortname"], doc["ctype"]
        if key in unique:
            assert unique[key]["sha256"] == record["sha256"]
            assert unique[key]["source_document"]["_id"] == doc["_id"]
        unique[key] = record
    assert len(unique) == 13 and {k[0] for k in unique} == COLLECTIONS
    planned = []
    for (collection, ctype), record in sorted(unique.items()):
        source = record["source_document"]
        assert sha(snapshot / record["file"]) == record["sha256"]
        data = decode(snapshot / record["file"], source)
        assert payload_sha(data) == record["sha256"]
        detname = "jungfrau" if collection == "jungfrau_000009" else "inline_alvium"
        info = manifest["detectors"][detname]
        metadata = {k: source[k] for k in ["dtype", "time_sec", "time_stamp", "version"] if k in source}
        metadata.update(experiment="prjdatpsana", run=1, run_end=1,
            detector=collection, shortname=collection, longname=info["uniqueid"],
            dettype=info["dettype"], detname=detname, ctype=ctype,
            run_orig=source.get("run_orig", source["run"]),
            tstamp_orig=source.get("time_stamp", ""),
            comment="Frozen calibration for synthetic MFX pilot; source events mfx101555026 run 10; valid only for prjdatpsana run 1.",
            extpars={"snapshot_manifest_sha256": MANIFEST_SHA,
                "source_db": record["source_db"], "source_document_id": source["_id"],
                "source_data_id": source["id_data"], "source_experiment": source["experiment"],
                "source_run": source["run"], "payload_sha256": record["sha256"],
                "source_events": {"experiment": "mfx101555026", "run": 10},
                "synthetic_dataset": True})
        planned.append({"collection": collection, "ctype": ctype,
            "file": record["file"], "sha256": record["sha256"],
            "bytes": record["bytes"], "metadata": metadata})
        del data
    # Before any write: no overlapping or conflicting documents allowed.
    destination = {c: docs(c) for c in sorted(COLLECTIONS)}
    assert all(not value for value in destination.values()), "Destination is not empty; inspect before proceeding"
    headers = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    privilege = requests.get(RW + DB + "/test_post_privilege", headers=headers).json()
    assert privilege.get("status") is True
    plan = {"snapshot_manifest_sha256": MANIFEST_SHA, "database": DB,
            "run": 1, "run_end": 1, "records": planned}
    if not args.deploy:
        output.mkdir(parents=True, exist_ok=False)
        save(output / "deployment_plan.json", plan)
        print(f"Preflight passed: {len(planned)} documents; {sum(x['bytes'] for x in planned)} bytes", flush=True)
        return
    assert plan == json.loads((output / "deployment_plan.json").read_text())
    assert not (output / "post_journal.json").exists(), "Prior write attempt exists; inspect journal before resuming"
    save(output / "preflight.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "privilege": privilege, "destination_before": destination})
    journal = []
    save(output / "deployment_journal.json", journal)
    write_enabled = True
    for item in planned:
        record = unique[(item["collection"], item["ctype"])]
        data = decode(snapshot / item["file"], record["source_document"])
        result = wu.add_data_and_doc(data, DB, item["collection"], **item["metadata"])
        assert result and all(result), f"Incomplete upload: {item['collection']}/{item['ctype']}"
        journal.append({"collection": item["collection"], "ctype": item["ctype"],
            "id_data": result[0], "id_doc": result[1], "sha256": item["sha256"]})
        save(output / "deployment_journal.json", journal)
        print(f"Deployed {item['collection']}/{item['ctype']}: {result[1]}", flush=True)
        del data
        gc.collect()
    write_enabled = False
    index = {(r["collection"], r["ctype"]): r for r in journal}
    selected = []
    for name, info in sorted(manifest["detectors"].items()):
        if not info["calibration_types"]:
            continue
        resolved = wu.calib_constants_all_types(info["uniqueid"], exp="prjdatpsana", run=1)
        assert set(resolved) == set(info["calibration_types"])
        for ctype, (data, document) in resolved.items():
            key = document["shortname"], ctype
            expected = index[key]
            assert document["_id"] == expected["id_doc"]
            assert document["id_data"] == expected["id_data"]
            assert document["experiment"] == "prjdatpsana"
            assert document["run"] == document["run_end"] == 1
            assert document["id_doc_exp"] == document["id_data_exp"] == 0
            source = unique[key]["source_document"]
            for field in ["data_type", "data_dtype", "data_shape"]:
                assert document.get(field) == source.get(field)
            assert payload_sha(data) == expected["sha256"]
            selected.append({"detector": name, "ctype": ctype, "id_doc": document["_id"],
                             "sha256": expected["sha256"], "destination_selected": True})
        del resolved, data
        gc.collect()
        print(f"Verified normal run-1 lookup for {name}", flush=True)
    after = {c: docs(c) for c in sorted(COLLECTIONS)}
    for collection, records in after.items():
        assert len(records) == sum(r["collection"] == collection for r in journal)
        assert {d["_id"] for d in records} == {r["id_doc"] for r in journal if r["collection"] == collection}
        # Experiment records are explicitly out of range for run 2. Normal
        # lookup there may still fall back to the shared detector DB.
        for ctype in {d["ctype"] for d in records}:
            assert wu.select_doc_in_run_range([d for d in records if d["ctype"] == ctype], 2) is None
    verification = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "database": DB, "run": 1, "run_end": 1, "document_count": len(journal),
        "payload_bytes": sum(x["bytes"] for x in planned), "selected": selected,
        "post_count": len(post_log), "write_urls": sorted({x["url"] for x in post_log}),
        "experiment_records_excluded_for_run2": True, "synthetic_xtc_created": False}
    save(output / "readback_verification.json", verification)
    print(f"Stage 2 verified: {len(journal)} documents, {len(selected)} detector/type selections", flush=True)


if __name__ == "__main__":
    main()

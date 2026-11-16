"""Collision checks and no-overwrite publication for the reviewed pilot only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from xtc_fixture import DEST, DEST_EXP, file_sha, require, save_json, sha


def preflight():
    import requests
    from krtc import KerberosTicket
    for folder in (DEST, DEST / "smalldata"):
        require(folder.is_dir() and os.access(folder, os.W_OK | os.X_OK),
                "Missing or unwritable destination directory: " + str(folder))
    collisions = [str(p) for folder in (DEST, DEST / "smalldata") for p in folder.glob("*r0001*.xtc2*")]
    require(not collisions, "Existing run-1 paths: " + repr(collisions))
    endpoint = f"https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/{DEST_EXP}/ws/"
    headers = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    responses = {}
    for label, suffix in (("runs", "runs"), ("files", "1/current_files_for_live_mode")):
        r = requests.get(endpoint + suffix, headers=headers, timeout=60)
        r.raise_for_status()
        result = r.json()
        require(result.get("success") is True and isinstance(result.get("value"), list), "Unexpected logbook response")
        responses[label] = result["value"]
    # This dedicated test experiment was empty at Stage 1. Fail closed if it
    # has acquired any run metadata; investigate rather than guess its schema.
    require(responses["runs"] == [] and responses["files"] == [], "Logbook no longer empty")
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "destination": str(DEST), "collisions": collisions, "logbook": responses,
            "available_bytes": os.statvfs(DEST).f_bavail * os.statvfs(DEST).f_frsize}


def publish(staging, verification, review_path, commit, journal):
    require(staging.resolve().parent == DEST.resolve() and staging.name.startswith(".synthetic-mfx-r0001-"), "Unexpected staging location")
    report = json.loads(verification.read_text())
    generation = json.loads((staging / "generation.json").read_text())
    review = json.loads(review_path.read_text())
    require(report["status"] == "pass" and report["generation_sha256"] == file_sha(staging / "generation.json"), "Missing matching validation")
    require(review["verdict"] == "pass" and review["reviewed_commit"] == commit and
            not any(f["blocking"] for f in review["findings"]), "Missing successful code review")
    # Require the implementation being used to be the exact reviewed tree.
    repo = Path(__file__).resolve().parents[2]
    reviewed_hashes = {}
    for name in ("xtc_fixture.py", "make_pilot.py", "verify_pilot.py", "pilot_admin.py"):
        p = Path(__file__).with_name(name)
        recorded = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":" + str(p.relative_to(repo))])
        require(recorded == p.read_bytes(), "Unreviewed implementation change")
        reviewed_hashes[name] = sha(recorded)
    for name in ("make_pilot.py", "xtc_fixture.py"):
        require(generation["code_sha256"].get(name) == reviewed_hashes[name], "Unreviewed generator provenance")
    for name in ("verify_pilot.py", "xtc_fixture.py"):
        require(report.get("code_sha256", {}).get(name) == reviewed_hashes[name], "Unreviewed validator provenance")
    checks = preflight()
    require(len(generation["streams"]) == len(report["streams"]) == 10, "Incomplete stream set")
    files = []
    for index, (stream, checked) in enumerate(zip(generation["streams"], report["streams"])):
        require(stream["stream"] == checked["stream"] == index and stream["files"] == checked["files"], "Validation manifest mismatch")
        for kind in ("bd", "smd"):
            item = stream["files"][kind]
            expected = f"{DEST_EXP}-r0001-s{index:03d}-c000" + (".xtc2" if kind == "bd" else ".smd.xtc2")
            relative = expected if kind == "bd" else "smalldata/" + expected
            require(item["relative_path"] == relative, "Unexpected publish path")
            source = staging / relative
            require(source.is_file() and not source.is_symlink() and source.stat().st_size == item["size"] and file_sha(source) == item["sha256"], "Staged bytes changed")
            files.append((kind, source, DEST / relative, item))
    # Publish bigdata first, then SMD. os.link atomically refuses collisions.
    # A failure leaves an explicit partial journal; never remove someone else's
    # paths or retry blindly. Complete-set publication is not atomic.
    files.sort(key=lambda row: row[0] == "smd")
    with journal.open("x") as log:
        def record(value):
            log.write(json.dumps(value) + "\n")
            log.flush()
            os.fsync(log.fileno())
        record({"phase": "preflight", "checks": checks, "reviewed_commit": commit})
        for kind, source, destination, item in files:
            record({"phase": "intent", "source": str(source), "destination": str(destination), "sha256": item["sha256"]})
            os.link(source, destination)
            folder = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(folder)
            finally:
                os.close(folder)
            require(destination.stat().st_ino == source.stat().st_ino, "Publication inode mismatch")
            record({"phase": "linked", "destination": str(destination), "sha256": item["sha256"]})
        record({"phase": "complete", "files": len(files), "completed_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--report", type=Path, required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--staging", type=Path, required=True)
    pub.add_argument("--verification", type=Path, required=True)
    pub.add_argument("--review", type=Path, required=True)
    pub.add_argument("--commit", required=True)
    pub.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        save_json(args.report, preflight())
    else:
        publish(args.staging, args.verification, args.review, args.commit, args.journal)

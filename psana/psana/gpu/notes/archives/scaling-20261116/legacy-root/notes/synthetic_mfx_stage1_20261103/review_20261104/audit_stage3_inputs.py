"""Read frozen SMD inputs and record Stage 3 construction prerequisites.

Requires the Stage 1 psana environment. Does not generate XTC or contact services.
"""
import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

from psana import dgram


SERVICES = {2: "Configure", 4: "BeginRun", 5: "EndRun", 6: "BeginStep",
            7: "EndStep", 8: "Enable", 9: "Disable"}


def audit(snapshot):
    manifest = json.loads((snapshot / "manifest.json").read_text())
    streams = []
    for source in manifest["streams"]:
        path = snapshot / "smd" / Path(source["smd"]["path"]).name
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        assert digest == source["smd"]["sha256"], path
        cfg = dgram.Dgram(view=blob, offset=0)
        names = sorted(k for k in vars(cfg.software) if not k.startswith("_"))
        assert "scan" not in names, names
        templates = {}
        beginruns, beginstep_sizes, enable_chunkinfo = [], [], []
        offset = 0
        while offset < len(blob):
            timestamp, env, src, damage, contains, extent = struct.unpack_from("<QIIHHI", blob, offset)
            size = 12 + extent
            assert extent >= 12 and offset + size <= len(blob)
            service = (env >> 24) & 15
            if service in SERVICES:
                record = {"smd_offset": offset, "size": size, "timestamp": timestamp,
                          "env": env, "src": src, "damage": damage,
                          "contains": contains, "extent": extent,
                          "header_hex": blob[offset:offset + 24].hex(),
                          "sha256": hashlib.sha256(blob[offset:offset + size]).hexdigest()}
                templates.setdefault(SERVICES[service], record)
            if service == 4:
                dg = dgram.Dgram(config=cfg, view=blob, offset=offset)
                for seg in dg.runinfo.values():
                    beginruns.append({"experiment": seg.runinfo.expt, "run": int(seg.runinfo.runnum)})
            elif service == 6:
                beginstep_sizes.append(size)
            elif service == 8:
                dg = dgram.Dgram(config=cfg, view=blob, offset=offset)
                enable_chunkinfo.append(hasattr(dg, "chunkinfo"))
            offset += size
        assert set(templates) == set(SERVICES.values())
        assert beginruns == [{"experiment": "mfx101555026", "run": 10}]
        assert beginstep_sizes and set(beginstep_sizes) == {24}
        assert enable_chunkinfo and not any(enable_chunkinfo)
        streams.append({"smd": str(path), "sha256": digest, "configure_names": names,
                        "beginrun": beginruns[0], "beginstep_sizes": beginstep_sizes,
                        "enable_has_chunkinfo": enable_chunkinfo, "templates": templates})
    assert len(streams) == 10
    return {"completed_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot": str(snapshot), "psana_dgram_module": dgram.__file__,
            "scope": "Frozen SMD prerequisites only; no generated files or EventBuilder test.",
            "streams": streams}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.snapshot)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print("Verified ten frozen SMD inputs and recorded per-stream transition templates.")

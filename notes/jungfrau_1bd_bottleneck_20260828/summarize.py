#!/usr/bin/env python3
"""Extract benchmark JSON records and normalize exact 1000-event payload rates."""

import argparse
import json
from pathlib import Path


MARKERS = (
    "READ_BASELINE",
    "KVIKIO_READ_BASELINE",
    "H2D_BASELINE",
    "CPU_1BD_PROFILE",
    "GPU_1BD_PROFILE",
)
EXACT_EVENT_BYTES = 33_557_176


def records(path):
    for line in path.read_text().replace("\x00", "").splitlines():
        for marker in MARKERS:
            prefix = marker + " "
            marker_at = line.find(prefix)
            if marker_at >= 0:
                row = json.loads(line[marker_at + len(prefix) :])
                row["marker"] = marker
                if marker in ("CPU_1BD_PROFILE", "GPU_1BD_PROFILE"):
                    row["exact_payload_gbps"] = (
                        row["events"] * EXACT_EVENT_BYTES / row["loop_s"] / 1e9
                    )
                yield row
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.logs:
        for row in records(path):
            print(json.dumps({"log": str(path), **row}, sort_keys=True))


if __name__ == "__main__":
    main()

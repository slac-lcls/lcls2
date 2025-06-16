import argparse
import sys
from typing import List

from psdaq.configdb.configdb_multimod import configdb_multimod


def change_gain_mode():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="jungfrau_gain_mode")
    parser.add_argument(
        "--cfg", type=str, default="BEAM", help="Config alias, e.g. BEAM, TEST, etc."
    )
    parser.add_argument(
        "--detname", type=str, default="jungfrau", help="Base detector name."
    )
    parser.add_argument("--hutch", type=str, default="mfx", help="Hutch/instrument.")
    parser.add_argument(
        "--mode",
        choices=["dynamic", "med", "low"],
        help="Gain mode to use. One of `dynamic`, `med`, or `low`",
        required=True,
    )
    parser.add_argument(
        "--nsegs",
        type=int,
        default=32,
        help="Number of jungfrau segments (all will be modified).",
    )
    args: argparse.Namespace = parser.parse_args()

    detector_segments: List[str] = []
    for seg in range(args.nsegs):
        detector_segments.append(f"{args.detname}_{seg}")

    new_gain_mode: int
    if args.mode == "dynamic":
        new_gain_mode = 0
    elif args.mode == "med":
        new_gain_mode = 3
    elif args.mode == "low":
        new_gain_mode = 4
    else:
        print("Unrecognized gain mode. Must be `dynamic`, `med`, or `low`")
        sys.exit(-1)

    configdb_multimod(
        URI_CONFIGDB="https://pswww.slac.stanford.edu/ws-auth/configdb/ws",
        DEV=args.cfg,
        INSTRUMENT=args.hutch,
        DETECTOR=detector_segments,
        CONFIG_KEY=["user","gainMode"],
        CONFIG_VALUE=new_gain_mode,
        MODIFY=True,
    )


if __name__ == "__main__":
    change_gain_mode()

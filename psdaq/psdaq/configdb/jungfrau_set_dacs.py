import argparse
import sys
from typing import List, Dict

from psdaq.configdb.configdb_multimod import configdb_multimod


def set_dacs():
    dacnames: List[str] = {
        "vb_ds",
        "vb_comp",
        "vb_pixbuf",
        "vref_ds",
        "vref_comp",
        "vref_prech",
        "vin_com",
        "vdd_prot",       
    }
    dacvals: List[int] = {
        1000,
        1220,
        750,
        480,
        420,
        1450,
        1053,
        3000,
    }

    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="jungfrau_gain_mode")
    parser.add_argument(
        "--cfg", type=str, default="BEAM", help="Config alias, e.g. BEAM, TEST, etc."
    )
    parser.add_argument(
        "--detname", type=str, default="jungfrau", help="Base detector name."
    )
    parser.add_argument("--hutch", type=str, default="mfx", help="Hutch/instrument.")
    parser.add_argument(
        "--dacs",
        type=int,
        nargs=8,
        metavar=("VB_DS", "VB_COMP", "VB_PIXBUF", "VREF_DS", "VREF_COMP", "VREF_PRECH", "VIN_COM", "VDD_PROT"),
        help="Value for the dac registers",
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

    if args.dacs is not None:
        dacvals = args.dacs

    dacs: Dict[int] = {k: v for k, v in zip(dacnames, dacvals)}       

    for dacname, dacvalue in dacs.items():
        configdb_multimod(
            URI_CONFIGDB="https://pswww.slac.stanford.edu/ws-auth/configdb/ws",
            DEV=args.cfg,
            INSTRUMENT=args.hutch,
            DETECTOR=detector_segments,
            CONFIG_KEY=["expert",dacname],
            CONFIG_VALUE=dacvalue,
            MODIFY=True,
        )


if __name__ == "__main__":
    set_dacs()

#!/usr/bin/env python3

"""Launch a PyDM GUI for ePixQuad standalone XpmMini file capture.

This opens two PyRogue roots against the same C1100 card:

1. lcls2-pgp-pcie-apps DevRoot in LCLS-II standalone XpmMini mode.
2. ePixQuad.Top with its StreamWriter connected to the camera data VCs.

The script does not write PROM contents or ASIC pixel maps.  The default
promWrEn=True only bypasses the ePixQuad startup PROM validation that fails on
the rdsrv421 test stand.  Do not execute ADC-training or PROM-writing commands
from the GUI unless that is intentional.
"""

import argparse
import os
import sys


def _parse_bool(text):
    return str(text).lower() in ("1", "true", "t", "yes", "y")


def _default_lcls2_yaml():
    submodule_dir = os.environ.get("SUBMODULEDIR")
    if not submodule_dir:
        return "config/defaults_LCLS-II.yml"
    return os.path.join(
        submodule_dir,
        "lcls2-pgp-pcie-apps",
        "software",
        "config",
        "defaults_LCLS-II.yml",
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Launch XpmMini timing plus ePixQuad StreamWriter PyDM GUI."
    )
    parser.add_argument("--dev", default="/dev/datadev_0", help="datadev device path")
    parser.add_argument("--lane", type=int, default=0, help="PGP lane connected to the camera")
    parser.add_argument("--data-vc", type=int, default=0, help="VC selected by AppLane.VcDataTap")
    parser.add_argument(
        "--camera-vc-mask",
        type=lambda value: int(value, 0),
        default=0xF,
        help="VC mask for ePixQuad.Top; default 0xf opens data/register/scope/monitoring",
    )
    parser.add_argument(
        "--prom-bypass",
        type=_parse_bool,
        default=True,
        help="pass promWrEn to ePixQuad.Top; default true skips startup PROM validation",
    )
    parser.add_argument(
        "--lcls2-yaml",
        default=_default_lcls2_yaml(),
        help="lcls2-pgp-pcie-apps LCLS-II defaults YAML loaded at DevRoot startup",
    )
    parser.add_argument(
        "--no-lcls2-yaml",
        action="store_true",
        help="do not load lcls2-pgp-pcie-apps LCLS-II defaults YAML",
    )
    parser.add_argument(
        "--start-run",
        action="store_true",
        help="execute DevRoot.StartRun() after both roots are open",
    )
    parser.add_argument(
        "--pollEn",
        type=_parse_bool,
        default=False,
        help="enable PyRogue polling",
    )
    parser.add_argument(
        "--initRead",
        type=_parse_bool,
        default=False,
        help="read all variables at root startup",
    )
    return parser.parse_args()


def _load_modules():
    try:
        from psdaq.utils import enable_lcls2_pgp_pcie_apps  # noqa: F401
        from psdaq.utils import enable_epix_quad1kfps  # noqa: F401
        import ePixQuad
        import lcls2_pgp_pcie_apps
        import pyrogue.pydm
    except Exception as exc:
        print("Failed to import DAQ/ePixQuad hardware modules.", file=sys.stderr)
        print("Source setup_env.sh first and confirm SUBMODULEDIR is set.", file=sys.stderr)
        print(f"SUBMODULEDIR={os.environ.get('SUBMODULEDIR', '<unset>')}", file=sys.stderr)
        print(f"Import error: {exc!r}", file=sys.stderr)
        raise SystemExit(1) from exc

    return ePixQuad, lcls2_pgp_pcie_apps, pyrogue.pydm


def _configure_app_lane(pbase, args):
    app = getattr(pbase.DevPcie.Application, f"AppLane[{args.lane}]")
    app.VcDataTap.set(args.data_vc)
    app.XpmPauseThresh.set(0x20)
    app.EventBuilder.Timeout.set(int(156.25e6 / 1080))
    app.EventBuilder.Bypass.set(0x0)
    app.EventBuilder.Blowoff.set(True)


def main():
    args = _parse_args()
    ePixQuad, lcls2_pgp_pcie_apps, pydm = _load_modules()
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xpmmini_writer_pydm.py")

    lcls2_yaml = None if args.no_lcls2_yaml else args.lcls2_yaml
    if lcls2_yaml is not None and not os.path.exists(lcls2_yaml):
        print(f"LCLS-II defaults YAML does not exist: {lcls2_yaml}", file=sys.stderr)
        return 1
    if not os.path.exists(ui_path):
        print(f"PyDM UI file does not exist: {ui_path}", file=sys.stderr)
        return 1

    pbase = lcls2_pgp_pcie_apps.DevRoot(
        dev=args.dev,
        enLclsI=False,
        enLclsII=True,
        yamlFileLclsI=None,
        yamlFileLclsII=lcls2_yaml,
        startupMode=True,
        standAloneMode=True,
        pgp4=True,
        dataVc=args.data_vc,
        pollEn=args.pollEn,
        initRead=args.initRead,
    )

    cbase = ePixQuad.Top(
        dev=args.dev,
        hwType="datadev",
        lane=args.lane,
        promWrEn=args.prom_bypass,
        enVcMask=args.camera_vc_mask,
        enWriter=True,
        enPrbs=False,
        pollEn=args.pollEn,
        initRead=args.initRead,
    )
    if hasattr(cbase, "StreamWriter"):
        cbase.StreamWriter.hidden = False

    run_started = False
    pbase_open = False
    cbase_open = False
    try:
        pbase.__enter__()
        pbase_open = True
        _configure_app_lane(pbase, args)
        cbase.__enter__()
        cbase_open = True

        if args.start_run:
            pbase.StartRun()
            run_started = True

        print()
        print("XpmMini/ePixQuad GUI is ready.")
        print(f"  C1100 server : {pbase.zmqServer.address}")
        print(f"  Camera server: {cbase.zmqServer.address}")
        print("  Camera writer: Top.StreamWriter")
        print()
        print("Typical capture flow:")
        print("  1. Run DevRoot.StartRun() unless --start-run was used.")
        print("  2. In the Camera System tab, set StreamWriter.DataFile, then click Open.")
        print("  3. Watch StreamWriter.FrameCount/TotalSize.")
        print("  4. Click Close, then DevRoot.StopRun().")
        print()

        pydm.runPyDM(
            serverList=f"{pbase.zmqServer.address},{cbase.zmqServer.address}",
            ui=ui_path,
            title="ePixQuad XpmMini Writer",
            sizeX=1000,
            sizeY=1000,
        )
    finally:
        if run_started:
            try:
                pbase.StopRun()
            except Exception:
                pass
        if cbase_open:
            cbase.__exit__(None, None, None)
        if pbase_open:
            pbase.__exit__(None, None, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

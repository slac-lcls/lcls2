#!/usr/bin/env python3

"""Diagnose standalone epixquad1kfps hardware access.

This script is intended for test stands such as the UED ePixQuad 1 kfps setup
on rdsrv421.  It does not start DAQ control, does not write ASIC pixel maps,
and does not write PROM contents.

The diagnostic performs four checks:

1. Read PGP4 lane status directly from the PCIe firmware.
2. Initialize the lcls2-pgp-pcie-apps C1100 application path, including the
   data VC tap used by the event builder.
3. Open the ePixQuad camera register tree on VC1 and report whether the
   startup PROM calibration check passes.  If that check fails, rerun this
   script in a fresh process with --force-prom-bypass to skip the startup PROM
   validation.  That mode does not write PROM unless PROM-writing commands are
   explicitly invoked elsewhere.
4. Probe ASIC SACI access.  The default probe only reads ASIC registers.  Any
   DAQ-like camera pre-init or write probe must be explicitly requested.
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace


def _parse_bool(text):
    return str(text).lower() in ("1", "true", "t", "yes", "y")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose C1100 PGP and ePixQuad 1 kfps camera register access."
    )
    parser.add_argument("--dev", default="/dev/datadev_0", help="datadev device path")
    parser.add_argument("--lane", type=int, default=0, help="PGP lane connected to the camera")
    parser.add_argument("--num-lanes", type=int, default=4, help="number of PGP lanes to inspect")
    parser.add_argument("--num-vc", type=int, default=4, help="number of virtual channels per lane")
    parser.add_argument("--data-vc", type=int, default=0, help="VC selected by AppLane.VcDataTap")
    parser.add_argument(
        "--camera-vc-mask",
        type=lambda value: int(value, 0),
        default=0x2,
        help="VC mask for ePixQuad.Top; default 0x2 opens VC1/SRP only",
    )
    parser.add_argument(
        "--standalone-timing",
        type=_parse_bool,
        default=True,
        help="use locally generated timing in C1100 DevRoot startup",
    )
    parser.add_argument(
        "--skip-pgp-status",
        action="store_true",
        help="skip direct PGP lane status reads",
    )
    parser.add_argument(
        "--skip-pcie-init",
        action="store_true",
        help="skip lcls2-pgp-pcie-apps DevRoot startup before camera access",
    )
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="skip ePixQuad camera register access",
    )
    parser.add_argument(
        "--no-prom-bypass-retry",
        action="store_true",
        help="deprecated; same-process PROM-bypass retry is disabled by default",
    )
    parser.add_argument(
        "--prom-bypass-retry",
        action="store_true",
        help="retry camera open with promWrEn=True in the same process after PROM check failure",
    )
    parser.add_argument(
        "--force-prom-bypass",
        action="store_true",
        help="open camera only with promWrEn=True, skipping startup PROM validation",
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="print full exception representations for failed register reads",
    )
    parser.add_argument(
        "--timing-sample-seconds",
        type=float,
        default=0.25,
        help="seconds between timing counter samples after C1100 pre-init",
    )
    parser.add_argument(
        "--skip-saci-probe",
        action="store_true",
        help="skip ASIC SACI register access probe after camera open",
    )
    parser.add_argument(
        "--saci-asic",
        type=int,
        default=0,
        help="ASIC index to use for the SACI probe",
    )
    parser.add_argument(
        "--saci-daq-like-preinit",
        action="store_true",
        help="write a minimal DAQ-like camera pre-init sequence before the SACI probe",
    )
    parser.add_argument(
        "--saci-write-probe",
        action="store_true",
        help="after read probes, write previous IsEn value back to test a SACI write",
    )
    parser.add_argument(
        "--saci-enable-probe",
        action="store_true",
        help="write IsEn=True, then restore the original IsEn value",
    )
    parser.add_argument(
        "--saci-writer-enable-probe",
        action="store_true",
        help="replicate the writer enable sequence: device enable=True, IsEn=True, then restore",
    )
    parser.add_argument(
        "--saci-compth-write-probe",
        action="store_true",
        help="write CompTH_DAC to --saci-compth-value, then restore the original value",
    )
    parser.add_argument(
        "--saci-compth-value",
        type=lambda value: int(value, 0),
        default=0x22,
        help="CompTH_DAC value for --saci-compth-write-probe; default 0x22 from the UED YAML",
    )
    return parser.parse_args()


def _load_hardware_modules():
    try:
        from psdaq.utils import enable_lcls2_pgp_pcie_apps  # noqa: F401
        from psdaq.utils import enable_epix_quad1kfps  # noqa: F401
        import axipcie
        import ePixQuad
        import lcls2_pgp_pcie_apps
        import pyrogue as pr
        import rogue.hardware.axi
        import surf.protocols.pgp as pgp
    except Exception as exc:
        print("Failed to import DAQ/ePixQuad hardware modules.", file=sys.stderr)
        print("Source the DAQ setup_env.sh first and confirm SUBMODULEDIR is set.", file=sys.stderr)
        print(f"SUBMODULEDIR={os.environ.get('SUBMODULEDIR', '<unset>')}", file=sys.stderr)
        print(f"Import error: {exc!r}", file=sys.stderr)
        raise SystemExit(1) from exc

    return SimpleNamespace(
        axipcie=axipcie,
        ePixQuad=ePixQuad,
        lcls2_pgp_pcie_apps=lcls2_pgp_pcie_apps,
        pr=pr,
        rogue=rogue,
        pgp=pgp,
    )


def _heading(text):
    print()
    print("=" * len(text))
    print(text)
    print("=" * len(text))


def _read_var(var, *, verbose_errors=False):
    try:
        return True, var.get(read=True)
    except Exception as exc:
        if verbose_errors:
            return False, f"ERR:{exc!r}"
        return False, f"ERR:{exc}"


def _print_read(label, var, *, verbose_errors=False, transform=None):
    ok, value = _read_var(var, verbose_errors=verbose_errors)
    if ok and transform is not None:
        value = transform(value)
    print(f"  {label:<22} {value}")
    return ok, value


def _write_var(label, var, value, *, verbose_errors=False):
    try:
        var.set(value)
    except Exception as exc:
        if verbose_errors:
            print(f"  {label:<22} ERR:{exc!r}")
        else:
            print(f"  {label:<22} ERR:{exc}")
        return False
    print(f"  {label:<22} OK")
    return True


def _mhz(raw):
    return f"{raw * 1.0e-6:.3f}"


def _is_true(value):
    return value is True or value == 1


def _check_pgp_status(mods, args):
    _heading("PGP4 lane status")

    class PgpRoot(mods.pr.Root):
        def __init__(self):
            super().__init__(
                name="pgpStatus",
                description="PGP status",
                pollEn=False,
                initRead=False,
            )
            self.memMap = mods.rogue.hardware.axi.AxiMemMap(args.dev)
            self.add(
                mods.axipcie.AxiPcieCore(
                    offset=0x00000000,
                    memBase=self.memMap,
                    numDmaLanes=args.num_lanes,
                    expand=False,
                )
            )
            for lane in range(args.num_lanes):
                self.add(
                    mods.pgp.Pgp4AxiL(
                        name=f"Lane{lane}",
                        offset=0x00800000 + lane * 0x00010000,
                        memBase=self.memMap,
                        numVc=args.num_vc,
                        writeEn=False,
                        expand=False,
                    )
                )

    linked_lanes = []
    with PgpRoot() as root:
        for lane in range(args.num_lanes):
            link = getattr(root, f"Lane{lane}")
            rx = link.RxStatus
            tx = link.TxStatus
            print(f"Lane {lane}:")
            _, rx_ready = _print_read(
                "RX LinkReady",
                rx.LinkReady,
                verbose_errors=args.verbose_errors,
            )
            _, rem_ready = _print_read(
                "RX RemRxLinkReady",
                rx.RemRxLinkReady,
                verbose_errors=args.verbose_errors,
            )
            _print_read("RX PhyRxActive", rx.PhyRxActive, verbose_errors=args.verbose_errors)
            _print_read("RX Clock MHz", rx.RxClockFreqRaw, verbose_errors=args.verbose_errors, transform=_mhz)
            _, tx_ready = _print_read(
                "TX LinkReady",
                tx.LinkReady,
                verbose_errors=args.verbose_errors,
            )
            _print_read("TX phyTxActive", tx.phyTxActive, verbose_errors=args.verbose_errors)
            _print_read("TX Clock MHz", tx.TxClockFreqRaw, verbose_errors=args.verbose_errors, transform=_mhz)
            _print_read("RX LinkDownCnt", rx.LinkDownCnt, verbose_errors=args.verbose_errors)
            _print_read("RX LinkErrorCnt", rx.LinkErrorCnt, verbose_errors=args.verbose_errors)
            if _is_true(rx_ready) and _is_true(rem_ready) and _is_true(tx_ready):
                linked_lanes.append(lane)

    if linked_lanes:
        print(f"Linked lane(s): {linked_lanes}")
    else:
        print("No fully linked lanes found.")
    return args.lane in linked_lanes


def _pcie_preinit(mods, args):
    _heading("C1100 application pre-init")
    pbase = mods.lcls2_pgp_pcie_apps.DevRoot(
        dev=args.dev,
        enLclsI=False,
        enLclsII=True,
        yamlFileLclsI=None,
        yamlFileLclsII=None,
        startupMode=True,
        standAloneMode=args.standalone_timing,
        pgp4=True,
        dataVc=args.data_vc,
        pollEn=False,
        initRead=False,
    )
    pbase.__enter__()

    app = getattr(pbase.DevPcie.Application, f"AppLane[{args.lane}]")
    app.VcDataTap.set(args.data_vc)
    app.XpmPauseThresh.set(0x20)
    app.EventBuilder.Timeout.set(int(156.25e6 / 1080))
    app.EventBuilder.Bypass.set(0x0)
    app.EventBuilder.Blowoff.set(True)
    time.sleep(0.1)

    print(f"Initialized DevRoot for {args.dev}, lane {args.lane}, data VC {args.data_vc}.")
    print(f"standAloneMode={args.standalone_timing}")
    _print_timing_status(pbase, args)
    return pbase


def _read_counter(var, args):
    ok, value = _read_var(var, verbose_errors=args.verbose_errors)
    if not ok:
        return value
    return value


def _print_timing_status(pbase, args):
    _heading("Timing receiver status")
    timing = pbase.DevPcie.Hsio.TimingRx.TimingFrameRx
    counter_names = (
        "sofCount",
        "eofCount",
        "FidCount",
        "CrcErrCount",
        "RxClkCount",
        "RxRstCount",
        "RxDecErrCount",
        "RxDspErrCount",
    )

    first = {name: _read_counter(getattr(timing, name), args) for name in counter_names}
    if args.timing_sample_seconds > 0:
        time.sleep(args.timing_sample_seconds)
    second = {name: _read_counter(getattr(timing, name), args) for name in counter_names}

    _print_read("RxLinkUp", timing.RxLinkUp, verbose_errors=args.verbose_errors)
    _print_read("RxDown", timing.RxDown, verbose_errors=args.verbose_errors)
    _print_read("ClkSel", timing.ClkSel, verbose_errors=args.verbose_errors)
    _print_read("ModeSel", timing.ModeSel, verbose_errors=args.verbose_errors)
    _print_read("ModeSelEn", timing.ModeSelEn, verbose_errors=args.verbose_errors)
    print(f"  sample seconds         {args.timing_sample_seconds:g}")

    for name in counter_names:
        value = second[name]
        delta = ""
        if isinstance(first[name], int) and isinstance(second[name], int):
            delta = f" delta={second[name] - first[name]}"
        print(f"  {name:<22} {value}{delta}")


def _print_camera_summary(cbase, args):
    _print_read(
        "Camera FW version",
        cbase.AxiVersion.FpgaVersion,
        verbose_errors=args.verbose_errors,
        transform=lambda value: f"0x{value:x}",
    )
    _print_read("Camera build", cbase.AxiVersion.BuildStamp, verbose_errors=args.verbose_errors)
    _print_read(
        "Camera GitHash",
        cbase.AxiVersion.GitHash,
        verbose_errors=args.verbose_errors,
        transform=lambda value: f"0x{value:x}",
    )
    for panel in range(4):
        _print_read(
            f"CarrierIdLow[{panel}]",
            cbase.SystemRegs.CarrierIdLow[panel],
            verbose_errors=args.verbose_errors,
            transform=lambda value: f"0x{value:08x}",
        )
        _print_read(
            f"CarrierIdHigh[{panel}]",
            cbase.SystemRegs.CarrierIdHigh[panel],
            verbose_errors=args.verbose_errors,
            transform=lambda value: f"0x{value:08x}",
        )
    _print_read("ADC test done", cbase.SystemRegs.AdcTestDone, verbose_errors=args.verbose_errors)
    _print_read("ADC test failed", cbase.SystemRegs.AdcTestFailed, verbose_errors=args.verbose_errors)


def _daq_like_camera_preinit(cbase, args):
    _heading("DAQ-like camera pre-init")
    print("This section writes volatile camera registers before the SACI probe.")

    writes = (
        ("SystemRegs.DcDcEnable", cbase.SystemRegs.DcDcEnable, 0xF),
        ("SystemRegs.AsicAnaEn", cbase.SystemRegs.AsicAnaEn, True),
        ("SystemRegs.AsicDigEn", cbase.SystemRegs.AsicDigEn, True),
        ("SystemRegs.AutoTrigEn", cbase.SystemRegs.AutoTrigEn, False),
        ("SystemRegs.TrigSrcSel", cbase.SystemRegs.TrigSrcSel, 0),
        ("SystemRegs.TrigEn", cbase.SystemRegs.TrigEn, True),
        ("RdoutCore.RdoutEn", cbase.RdoutCore.RdoutEn, True),
        ("AcqCore.AsicPpmatForce", cbase.AcqCore.AsicPpmatForce, True),
        ("AcqCore.AsicPpmatValue", cbase.AcqCore.AsicPpmatValue, True),
        ("AcqCore.AsicRoClkT", cbase.AcqCore.AsicRoClkT, 0xAAAA0003),
        ("RdoutCore.AdcPipelineDelay", cbase.RdoutCore.AdcPipelineDelay, 0xAAAA003D),
    )

    ok = True
    for label, var, value in writes:
        ok &= _write_var(label, var, value, verbose_errors=args.verbose_errors)
        if label == "SystemRegs.AsicDigEn":
            time.sleep(0.1)

    adc_done_ok, adc_done = _print_read(
        "ADC test done",
        cbase.SystemRegs.AdcTestDone,
        verbose_errors=args.verbose_errors,
    )
    adc_failed_ok, adc_failed = _print_read(
        "ADC test failed",
        cbase.SystemRegs.AdcTestFailed,
        verbose_errors=args.verbose_errors,
    )

    if adc_failed_ok and _is_true(adc_failed):
        print("  ADC restart           requested")
        ok &= _write_var("SystemRegs.AdcReqStart", cbase.SystemRegs.AdcReqStart, True, verbose_errors=args.verbose_errors)
        time.sleep(1.0e-6)
        ok &= _write_var("SystemRegs.AdcReqStart", cbase.SystemRegs.AdcReqStart, False, verbose_errors=args.verbose_errors)

    if not (adc_done_ok and _is_true(adc_done)):
        for _ in range(1000):
            time.sleep(0.001)
            done_ok, done = _read_var(cbase.SystemRegs.AdcTestDone, verbose_errors=args.verbose_errors)
            if done_ok and _is_true(done):
                print("  ADC wait              done")
                break
        else:
            print("  ADC wait              timed out")

    return ok


def _check_saci_access(cbase, args):
    if args.skip_saci_probe:
        return True

    _heading("ASIC SACI register access")
    print(f"ASIC index: {args.saci_asic}")
    preinit_ok = True
    if args.saci_daq_like_preinit:
        preinit_ok = _daq_like_camera_preinit(cbase, args)

    try:
        saci = cbase.Epix10kaSaci[args.saci_asic]
    except Exception as exc:
        print(f"  Epix10kaSaci lookup   ERR:{exc!r}" if args.verbose_errors else f"  Epix10kaSaci lookup   ERR:{exc}")
        return False

    probes = (
        ("CompTH_DAC", saci.CompTH_DAC),
        ("TestBe", saci.TestBe),
        ("IsEn", saci.IsEn),
        ("trbit", saci.trbit),
    )

    ok = preinit_ok
    values = {}
    for label, var in probes:
        read_ok, value = _print_read(label, var, verbose_errors=args.verbose_errors)
        values[label] = value
        ok &= read_ok

    if args.saci_write_probe:
        print()
        print("SACI write probe: writing the previously read IsEn value back.")
        if "IsEn" in values and not isinstance(values["IsEn"], str):
            ok &= _write_var("IsEn write-back", saci.IsEn, bool(values["IsEn"]), verbose_errors=args.verbose_errors)
        else:
            print("  IsEn write-back       skipped because IsEn read failed")
            ok = False

    if args.saci_enable_probe:
        print()
        print("SACI enable probe: writing IsEn=True, then restoring the original value.")
        if "IsEn" in values and not isinstance(values["IsEn"], str):
            original_is_en = bool(values["IsEn"])
            enable_ok = _write_var("IsEn=True", saci.IsEn, True, verbose_errors=args.verbose_errors)
            restore_ok = _write_var("IsEn restore", saci.IsEn, original_is_en, verbose_errors=args.verbose_errors)
            ok &= enable_ok and restore_ok
        else:
            print("  IsEn=True             skipped because IsEn read failed")
            ok = False

    if args.saci_writer_enable_probe:
        print()
        print("SACI writer-enable probe: device enable=True, IsEn=True, then restore.")
        enable_read_ok, original_enable = _read_var(saci.enable, verbose_errors=args.verbose_errors)
        original_is_en_ok = "IsEn" in values and not isinstance(values["IsEn"], str)
        if not enable_read_ok:
            print(f"  device enable read    {original_enable}")
        if original_is_en_ok:
            original_is_en = bool(values["IsEn"])
        else:
            print("  IsEn=True             skipped because IsEn read failed")

        device_enable_ok = _write_var(
            "device enable=True",
            saci.enable,
            True,
            verbose_errors=args.verbose_errors,
        )
        is_en_ok = False
        if original_is_en_ok:
            is_en_ok = _write_var("IsEn=True", saci.IsEn, True, verbose_errors=args.verbose_errors)

        restore_ok = True
        if original_is_en_ok:
            restore_ok &= _write_var(
                "IsEn restore",
                saci.IsEn,
                original_is_en,
                verbose_errors=args.verbose_errors,
            )
        if enable_read_ok:
            restore_ok &= _write_var(
                "device enable restore",
                saci.enable,
                bool(original_enable),
                verbose_errors=args.verbose_errors,
            )
        ok &= device_enable_ok and is_en_ok and restore_ok

    if args.saci_compth_write_probe:
        print()
        print(
            "SACI CompTH_DAC probe: writing 0x%x, then restoring the original value."
            % args.saci_compth_value
        )
        if "CompTH_DAC" in values and not isinstance(values["CompTH_DAC"], str):
            original_compth = int(values["CompTH_DAC"])
            compth_ok = _write_var(
                "CompTH_DAC write",
                saci.CompTH_DAC,
                int(args.saci_compth_value),
                verbose_errors=args.verbose_errors,
            )
            restore_ok = _write_var(
                "CompTH_DAC restore",
                saci.CompTH_DAC,
                original_compth,
                verbose_errors=args.verbose_errors,
            )
            ok &= compth_ok and restore_ok
        else:
            print("  CompTH_DAC write      skipped because CompTH_DAC read failed")
            ok = False

    if ok:
        print("ASIC SACI probe succeeded.")
    else:
        print("ASIC SACI probe failed.")
    return ok


def _try_camera_open(mods, args, *, prom_bypass):
    mode = "promWrEn=True, PROM validation skipped" if prom_bypass else "promWrEn=False, PROM validation enabled"
    print(f"Opening camera with {mode}")
    cbase = mods.ePixQuad.Top(
        dev=args.dev,
        hwType="datadev",
        lane=args.lane,
        promWrEn=prom_bypass,
        enVcMask=args.camera_vc_mask,
        enWriter=False,
        enPrbs=False,
        pollEn=False,
        initRead=False,
    )
    try:
        cbase.__enter__()
    except Exception as exc:
        print("Camera open failed.")
        if args.verbose_errors:
            print(f"  {exc!r}")
        else:
            print(f"  {exc}")
        try:
            cbase.__exit__(None, None, None)
        except Exception:
            pass
        return False

    try:
        print("Camera register access succeeded.")
        _print_camera_summary(cbase, args)
        return _check_saci_access(cbase, args)
    finally:
        cbase.__exit__(None, None, None)


def _check_camera(mods, args):
    _heading("Camera VC1 register access")
    if args.force_prom_bypass:
        return _try_camera_open(mods, args, prom_bypass=True)

    normal_ok = _try_camera_open(mods, args, prom_bypass=False)
    if normal_ok or args.no_prom_bypass_retry or not args.prom_bypass_retry:
        if not normal_ok and not args.prom_bypass_retry:
            print()
            print("Skipping same-process PROM-bypass retry.")
            print("A failed ePixQuad Top.start() can leave VC1 open until the process exits.")
            print("To bypass only the startup PROM validation, rerun in a fresh process:")
            print(f"  python {sys.argv[0]} --dev {args.dev} --lane {args.lane} --force-prom-bypass")
        return normal_ok

    print()
    print("Retrying with promWrEn=True to bypass startup PROM validation.")
    print("This diagnostic still does not invoke ADC training or PROM-writing commands.")
    return _try_camera_open(mods, args, prom_bypass=True)


def main():
    args = _parse_args()
    mods = _load_hardware_modules()

    print("epixquad1kfps standalone access diagnostic")
    print(f"dev={args.dev} lane={args.lane} SUBMODULEDIR={os.environ.get('SUBMODULEDIR', '<unset>')}")

    if not args.skip_pgp_status:
        _check_pgp_status(mods, args)

    pbase = None
    camera_ok = True
    try:
        if not args.skip_pcie_init:
            pbase = _pcie_preinit(mods, args)
        if not args.skip_camera:
            camera_ok = _check_camera(mods, args)
    finally:
        if pbase is not None:
            pbase.__exit__(None, None, None)

    if not camera_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

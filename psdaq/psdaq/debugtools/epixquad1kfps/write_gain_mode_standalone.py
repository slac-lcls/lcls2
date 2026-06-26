#!/usr/bin/env python3

"""Program ePixQuad1kfps fixed-gain test modes without DAQ or a GUI.

This opens the C1100 DevRoot in standalone XpmMini mode and the ePixQuad camera
register VC, then writes only the ASIC gain-map registers:

  FH      all pixels 0xc, trbit=1
  FM      all pixels 0xc, trbit=0
  FL      all pixels 0x8, trbit=0
  MapFML  FM background, selected FL pixels
  MapFHL  FH background, selected FL pixels

The matrix write follows psdaq.configdb.epixquad1kfps_config:
PrepareMultiConfig + WriteMatrixData for the full ASIC background, then
RowCounter/ColCounter/WritePixelData for selected pixels.
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PSDAQ_PACKAGE_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PSDAQ_PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PSDAQ_PACKAGE_PARENT)

from psdaq.configdb.epixquad_layout import RAW_ASIC_LAYOUT, RAW_SHAPE


BANK_OFFSETS = ((0xE << 7), (0xD << 7), (0xB << 7), (0x7 << 7))
ASIC_COUNT = 16
ASIC_ROWS = 178
ASIC_COLS = 192


class GainMode:
    def __init__(self, name, background_value, trbit, selected_value, description):
        self.name = name
        self.background_value = background_value
        self.trbit = trbit
        self.selected_value = selected_value
        self.description = description


GAIN_MODES = {
    "fh": GainMode("FH", 0xC, 1, None, "fixed high on every pixel"),
    "fm": GainMode("FM", 0xC, 0, None, "fixed medium on every pixel"),
    "fl": GainMode("FL", 0x8, 0, None, "fixed low on every pixel"),
    "mapfml": GainMode("MapFML", 0xC, 0, 0x8, "FM background with selected FL pixels"),
    "mapfhl": GainMode("MapFHL", 0xC, 1, 0x8, "FH background with selected FL pixels"),
}


DEFAULT_SELECTED_PIXELS = (
    # ASIC-local coordinates.  These stay away from edge/ghost rows and touch
    # all four banks on the first segment.
    (0, 12, 7),
    (0, 12, 55),
    (0, 145, 103),
    (0, 145, 151),
    (1, 12, 7),
    (1, 12, 55),
    (1, 145, 103),
    (1, 145, 151),
    (2, 12, 7),
    (2, 12, 55),
    (2, 145, 103),
    (2, 145, 151),
    (3, 12, 7),
    (3, 12, 55),
    (3, 145, 103),
    (3, 145, 151),
)


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


def _default_ued_camera_yaml():
    submodule_dir = os.environ.get("SUBMODULEDIR")
    if not submodule_dir:
        return None
    return os.path.join(
        submodule_dir,
        "epix-quad-1kfps",
        "software",
        "yml",
        "ued",
        "epixQuad_ASICs_allAsics_UED_1080Hz_settings.yml",
    )


def _parse_asics(text):
    out = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_text, hi_text = item.split("-", 1)
            lo = int(lo_text)
            hi = int(hi_text)
            if lo > hi:
                raise argparse.ArgumentTypeError(f"invalid ASIC range {item!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(item))
    bad = sorted(i for i in out if i < 0 or i >= ASIC_COUNT)
    if bad:
        raise argparse.ArgumentTypeError(f"ASIC indices out of range 0-15: {bad}")
    return sorted(out)


def _parse_triplet(text, label):
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{label} expects A,B,C, got {text!r}")
    try:
        return tuple(int(p, 0) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} has a non-integer field: {text!r}") from exc


def _parse_pixel(text):
    asic, row, col = _parse_triplet(text, "--pixel")
    _validate_direct_pixel(asic, row, col)
    return asic, row, col


def _parse_raw_pixel(text):
    seg, row, col = _parse_triplet(text, "--raw-pixel")
    if not (0 <= seg < RAW_SHAPE[0]):
        raise argparse.ArgumentTypeError(f"raw segment out of range 0-{RAW_SHAPE[0] - 1}: {seg}")
    if not (0 <= row < RAW_SHAPE[1]):
        raise argparse.ArgumentTypeError(f"raw row out of range 0-{RAW_SHAPE[1] - 1}: {row}")
    if not (0 <= col < RAW_SHAPE[2]):
        raise argparse.ArgumentTypeError(f"raw col out of range 0-{RAW_SHAPE[2] - 1}: {col}")
    return seg, row, col


def _validate_direct_pixel(asic, row, col):
    if not (0 <= asic < ASIC_COUNT):
        raise argparse.ArgumentTypeError(f"ASIC out of range 0-15: {asic}")
    if not (0 <= row < ASIC_ROWS):
        raise argparse.ArgumentTypeError(f"ASIC row out of range 0-{ASIC_ROWS - 1}: {row}")
    if not (0 <= col < ASIC_COLS):
        raise argparse.ArgumentTypeError(f"ASIC col out of range 0-{ASIC_COLS - 1}: {col}")


def _normalize_mode(text):
    key = str(text).lower()
    if key not in GAIN_MODES:
        choices = ", ".join(mode.name for mode in GAIN_MODES.values())
        raise argparse.ArgumentTypeError(f"mode must be one of: {choices}")
    return GAIN_MODES[key]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Write ePixQuad1kfps FH/FM/FL/MapFML/MapFHL gain test modes."
    )
    parser.add_argument("--mode", required=True, type=_normalize_mode)
    parser.add_argument("--dev", default="/dev/datadev_0", help="datadev device path")
    parser.add_argument("--lane", type=int, default=0, help="PGP lane connected to the camera")
    parser.add_argument("--data-vc", type=int, default=0, help="C1100 AppLane.VcDataTap value")
    parser.add_argument(
        "--camera-vc-mask",
        type=lambda value: int(value, 0),
        default=0x2,
        help="ePixQuad.Top VC mask; default 0x2 opens only camera register VC1",
    )
    parser.add_argument(
        "--prom-bypass",
        type=_parse_bool,
        default=True,
        help="pass promWrEn to ePixQuad.Top; default true skips startup PROM validation",
    )
    parser.add_argument(
        "--asics",
        type=_parse_asics,
        default=list(range(ASIC_COUNT)),
        help="ASICs to write, e.g. 0-15 or 0,1,2,3; default all",
    )
    parser.add_argument(
        "--pixel",
        action="append",
        type=_parse_pixel,
        default=[],
        metavar="ASIC,ROW,COL",
        help="selected FL pixel in ASIC-local coordinates; repeatable",
    )
    parser.add_argument(
        "--raw-pixel",
        action="append",
        type=_parse_raw_pixel,
        default=[],
        metavar="SEG,ROW,COL",
        help="selected FL pixel in raw-view segment coordinates; repeatable",
    )
    parser.add_argument(
        "--no-default-pixels",
        action="store_true",
        help="for Map modes, do not use the built-in selected-pixel list",
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
        "--camera-yaml",
        default=None,
        help="optional ePixQuad YAML to LoadConfig before writing the gain mode",
    )
    parser.add_argument(
        "--load-ued-yaml",
        action="store_true",
        help="load the standard UED ePixQuad YAML before writing the gain mode",
    )
    parser.add_argument(
        "--save-expected-map",
        default=None,
        help="optional .npy path for expected raw-view FL mask, shape (4,352,384)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the register-write plan without opening hardware",
    )
    parser.add_argument(
        "--leave-trigger-off",
        action="store_true",
        help="leave camera trigger/readout disabled after writing",
    )
    parser.add_argument("--verbose", action="store_true", help="print extra progress")
    return parser.parse_args()


def _load_modules():
    try:
        from psdaq.utils import enable_lcls2_pgp_pcie_apps  # noqa: F401
        from psdaq.utils import enable_epix_quad1kfps  # noqa: F401
        import ePixQuad
        import lcls2_pgp_pcie_apps
    except Exception as exc:
        print("Failed to import DAQ/ePixQuad hardware modules.", file=sys.stderr)
        print("Source setup_env.sh first and confirm SUBMODULEDIR is set.", file=sys.stderr)
        print(f"SUBMODULEDIR={os.environ.get('SUBMODULEDIR', '<unset>')}", file=sys.stderr)
        print(f"Import error: {exc!r}", file=sys.stderr)
        raise SystemExit(1) from exc
    return ePixQuad, lcls2_pgp_pcie_apps


def _raw_to_direct(seg, row, col):
    for layout in RAW_ASIC_LAYOUT:
        r0, r1 = layout["row_slice"]
        c0, c1 = layout["col_slice"]
        if not (r0 <= row < r1 and c0 <= col < c1):
            continue

        raw_local_row = row - r0
        raw_local_col = col - c0
        if layout["operator"] == "identity":
            prog_row = raw_local_row
            prog_col = raw_local_col
        elif layout["operator"] == "rot180":
            prog_row = (r1 - r0 - 1) - raw_local_row
            prog_col = (c1 - c0 - 1) - raw_local_col
        else:
            raise ValueError(f"unsupported raw-map operator {layout['operator']!r}")

        asic = 4 * seg + int(layout["slot"])
        return asic, int(prog_row), int(prog_col)

    raise ValueError(
        f"raw pixel seg={seg} row={row} col={col} does not fall in a mapped ASIC tile"
    )


def _direct_to_raw(asic, row, col):
    if row >= RAW_SHAPE[1] // 2:
        return None

    seg = asic // 4
    slot = asic % 4
    layout = next(layout for layout in RAW_ASIC_LAYOUT if layout["slot"] == slot)
    r0, r1 = layout["row_slice"]
    c0, c1 = layout["col_slice"]
    if layout["operator"] == "identity":
        raw_local_row = row
        raw_local_col = col
    elif layout["operator"] == "rot180":
        raw_local_row = (r1 - r0 - 1) - row
        raw_local_col = (c1 - c0 - 1) - col
    else:
        raise ValueError(f"unsupported raw-map operator {layout['operator']!r}")

    if not (0 <= raw_local_row < (r1 - r0) and 0 <= raw_local_col < (c1 - c0)):
        return None
    return seg, int(r0 + raw_local_row), int(c0 + raw_local_col)


def _raw_to_tiled(raw_pixel):
    seg, row, col = raw_pixel
    if seg == 3:
        return row, col
    if seg == 2:
        return row, col + RAW_SHAPE[2]
    if seg == 1:
        return row + RAW_SHAPE[1], col
    if seg == 0:
        return row + RAW_SHAPE[1], col + RAW_SHAPE[2]
    raise ValueError(f"unexpected segment {seg}")


def _selected_pixels(args):
    selected = []
    if args.mode.selected_value is None:
        if args.pixel or args.raw_pixel:
            raise ValueError(f"{args.mode.name} is a full-matrix mode; selected pixels are only used in Map modes")
        return selected

    if not args.no_default_pixels and not args.pixel and not args.raw_pixel:
        selected.extend(DEFAULT_SELECTED_PIXELS)

    selected.extend(args.pixel)
    for raw_pixel in args.raw_pixel:
        selected.append(_raw_to_direct(*raw_pixel))

    selected = list(dict.fromkeys(selected))
    outside = [pix for pix in selected if pix[0] not in args.asics]
    if outside:
        raise ValueError(
            "selected pixels map to ASICs outside --asics: "
            + ", ".join(f"a{a}:r{r}:c{c}" for a, r, c in outside[:8])
        )
    return selected


def _expected_raw_map(mode, selected):
    out = np.zeros(RAW_SHAPE, dtype=np.uint8)
    if mode.name == "FL":
        out.fill(1)
        return out
    if mode.selected_value is None:
        return out

    for asic, row, col in selected:
        raw_pixel = _direct_to_raw(asic, row, col)
        if raw_pixel is not None:
            out[raw_pixel] = 1
    return out


def _print_plan(args, selected):
    mode = args.mode
    print("ePixQuad1kfps standalone gain-mode write plan")
    print(f"  mode              : {mode.name} ({mode.description})")
    print(f"  dev/lane          : {args.dev} lane {args.lane}")
    print(f"  asics             : {args.asics}")
    print(f"  background value  : {mode.background_value} (0x{mode.background_value:x})")
    print(f"  trbit             : {mode.trbit}")
    if mode.selected_value is not None:
        print(f"  selected value    : {mode.selected_value} (0x{mode.selected_value:x})")
        print(f"  selected pixels   : {len(selected)}")
        for asic, row, col in selected[:24]:
            bank = col // 48
            bank_col = col % 48
            raw_pixel = _direct_to_raw(asic, row, col)
            raw_text = "raw=unmapped"
            tile_text = ""
            if raw_pixel is not None:
                raw_text = "raw=s%d,r%d,c%d" % raw_pixel
                tile_row, tile_col = _raw_to_tiled(raw_pixel)
                tile_text = f" tiled=r{tile_row},c{tile_col}"
            print(
                f"    asic={asic:2d} row={row:3d} col={col:3d} "
                f"bank={bank} bank_col={bank_col:2d} {raw_text}{tile_text}"
            )
        if len(selected) > 24:
            print(f"    ... {len(selected) - 24} more")
    else:
        print("  selected pixels   : none")
    if args.camera_yaml:
        print(f"  camera YAML       : {args.camera_yaml}")
    elif args.load_ued_yaml:
        print(f"  camera YAML       : {_default_ued_camera_yaml()}")
    print(f"  dry run           : {args.dry_run}")


def _configure_app_lane(pbase, args):
    app = getattr(pbase.DevPcie.Application, f"AppLane[{args.lane}]")
    app.VcDataTap.set(args.data_vc)
    app.XpmPauseThresh.set(0x20)
    app.EventBuilder.Timeout.set(int(156.25e6 / 1080))
    app.EventBuilder.Bypass.set(0x0)
    app.EventBuilder.Blowoff.set(True)
    try:
        pbase.RunState.set(False)
    except Exception:
        pass


def _safe_get(variable):
    try:
        return variable.get(read=True)
    except Exception:
        try:
            return variable.get()
        except Exception:
            return None


def _safe_set(variable, value, label):
    try:
        variable.set(value)
    except Exception as exc:
        raise RuntimeError(f"failed setting {label} to {value!r}: {exc}") from exc


def _disable_camera_trigger(cbase):
    state = {
        "TrigEn": _safe_get(cbase.SystemRegs.TrigEn),
        "AutoTrigEn": _safe_get(cbase.SystemRegs.AutoTrigEn),
        "RdoutEn": _safe_get(cbase.RdoutCore.RdoutEn),
    }
    _safe_set(cbase.SystemRegs.TrigEn, False, "SystemRegs.TrigEn")
    _safe_set(cbase.SystemRegs.AutoTrigEn, False, "SystemRegs.AutoTrigEn")
    _safe_set(cbase.RdoutCore.RdoutEn, False, "RdoutCore.RdoutEn")
    return state


def _restore_camera_trigger(cbase, state):
    if state.get("RdoutEn") is not None:
        _safe_set(cbase.RdoutCore.RdoutEn, state["RdoutEn"], "RdoutCore.RdoutEn")
    if state.get("AutoTrigEn") is not None:
        _safe_set(cbase.SystemRegs.AutoTrigEn, state["AutoTrigEn"], "SystemRegs.AutoTrigEn")
    if state.get("TrigEn") is not None:
        _safe_set(cbase.SystemRegs.TrigEn, state["TrigEn"], "SystemRegs.TrigEn")


def _write_gain_mode(cbase, mode, asics, selected, verbose=False):
    t0 = time.perf_counter()
    for asic in asics:
        saci = cbase.Epix10kaSaci[asic]
        if verbose:
            print(f"  enable ASIC {asic}")
        _safe_set(saci.enable, True, f"Epix10kaSaci[{asic}].enable")
        _safe_set(saci.IsEn, True, f"Epix10kaSaci[{asic}].IsEn")

    try:
        for asic in asics:
            saci = cbase.Epix10kaSaci[asic]
            _safe_set(saci.trbit, int(mode.trbit), f"Epix10kaSaci[{asic}].trbit")

        for asic in asics:
            saci = cbase.Epix10kaSaci[asic]
            _safe_set(saci.PrepareMultiConfig, 0, f"Epix10kaSaci[{asic}].PrepareMultiConfig")
            _safe_set(saci.WriteMatrixData, int(mode.background_value), f"Epix10kaSaci[{asic}].WriteMatrixData")

        for asic, row, col in selected:
            bank = int(col) // 48
            bank_col = int(col) % 48
            saci = cbase.Epix10kaSaci[int(asic)]
            _safe_set(saci.RowCounter, int(row), f"Epix10kaSaci[{asic}].RowCounter")
            _safe_set(saci.ColCounter, BANK_OFFSETS[bank] | bank_col, f"Epix10kaSaci[{asic}].ColCounter")
            _safe_set(saci.WritePixelData, int(mode.selected_value), f"Epix10kaSaci[{asic}].WritePixelData")
    finally:
        for asic in asics:
            saci = cbase.Epix10kaSaci[asic]
            try:
                saci.IsEn.set(False)
                saci.enable.set(False)
            except Exception as exc:
                print(f"Warning: failed disabling ASIC {asic}: {exc}", file=sys.stderr)

    print(
        "Gain write complete: mode=%s asics=%d selected_pixels=%d elapsed=%.3fs"
        % (mode.name, len(asics), len(selected), time.perf_counter() - t0)
    )


def _open_roots(args):
    ePixQuad, lcls2_pgp_pcie_apps = _load_modules()
    lcls2_yaml = None if args.no_lcls2_yaml else args.lcls2_yaml
    if lcls2_yaml is not None and not os.path.exists(lcls2_yaml):
        raise FileNotFoundError(f"LCLS-II defaults YAML does not exist: {lcls2_yaml}")

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
        pollEn=False,
        initRead=False,
    )
    cbase = ePixQuad.Top(
        dev=args.dev,
        hwType="datadev",
        lane=args.lane,
        promWrEn=args.prom_bypass,
        enVcMask=args.camera_vc_mask,
        enWriter=False,
        enPrbs=False,
        pollEn=False,
        initRead=False,
    )
    return pbase, cbase


def _load_camera_yaml(cbase, args):
    yaml_path = args.camera_yaml
    if args.load_ued_yaml:
        if yaml_path is None:
            yaml_path = _default_ued_camera_yaml()
        if yaml_path is None:
            raise RuntimeError("SUBMODULEDIR is not set; cannot locate the default UED YAML")
    if not yaml_path:
        return
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"camera YAML does not exist: {yaml_path}")
    print(f"Loading camera YAML: {yaml_path}")
    cbase.LoadConfig(yaml_path)
    print("Camera YAML loaded")


def main():
    args = _parse_args()
    selected = _selected_pixels(args)
    _print_plan(args, selected)

    if args.save_expected_map:
        expected = _expected_raw_map(args.mode, selected)
        np.save(args.save_expected_map, expected)
        print(f"Saved expected raw-view FL mask: {args.save_expected_map}")

    if args.dry_run:
        return 0

    pbase = None
    cbase = None
    pbase_open = False
    cbase_open = False
    trigger_state = None
    try:
        pbase, cbase = _open_roots(args)
        pbase.__enter__()
        pbase_open = True
        _configure_app_lane(pbase, args)

        cbase.__enter__()
        cbase_open = True
        trigger_state = _disable_camera_trigger(cbase)
        _load_camera_yaml(cbase, args)

        # LoadConfig can write trigger fields, so force them off again before
        # touching the ASIC pixel matrix.
        _disable_camera_trigger(cbase)
        _write_gain_mode(cbase, args.mode, args.asics, selected, verbose=args.verbose)

        if not args.leave_trigger_off:
            _restore_camera_trigger(cbase, trigger_state)
            trigger_state = None
            print("Restored previous camera trigger/readout state")
        else:
            print("Left camera trigger/readout disabled")
    finally:
        if trigger_state is not None and cbase_open and not args.leave_trigger_off:
            try:
                _restore_camera_trigger(cbase, trigger_state)
            except Exception as exc:
                print(f"Warning: failed restoring trigger state: {exc}", file=sys.stderr)
        if cbase_open:
            cbase.__exit__(None, None, None)
        if pbase_open:
            pbase.__exit__(None, None, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

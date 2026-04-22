from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.cas.xpm_utils import timTxId
from .xpmmini import *
from psdaq.utils import enable_epix_quad1kfps
import ePixQuad
from psdaq.utils import enable_lcls2_pgp_pcie_apps
import lcls2_pgp_pcie_apps
import rogue
#import epix
import time
import json
import os
import numpy as np
import IPython
from collections import deque
import surf.protocols.batcher  as batcher  # for Start/StopRun
import l2si_core               as l2si
import lcls2_pgp_fw_lib.shared as shared
import logging
from psdaq.debugtools.epixquad1kfps.pattern_loader import load_debug_override

base = None
pv = None
lane = 0
chan = None
group = None
ocfg = None
segids = None
seglist = [0,1,2,3,4]
debug_override = None

DEBUG_PIXEL_MASK_SAVED=False
DEBUG_ADC_TRAIN_WRITE=False
DEBUG_RANDOM_PIXEL_MAP=False
USE_ACCELERATED_MATRIX_WRITE=False
BANK_OFFSETS = ((0xe<<7),(0xd<<7),(0xb<<7),(0x7<<7))

RAW_MASK_ASIC_LAYOUT = (
    {'slot': 0, 'row_slice': (176, 352), 'col_slice': (192, 384), 'operator': 'identity'},
    {'slot': 1, 'row_slice': (0, 176),   'col_slice': (192, 384), 'operator': 'rot180'},
    {'slot': 2, 'row_slice': (0, 176),   'col_slice': (0, 192),   'operator': 'rot180'},
    {'slot': 3, 'row_slice': (176, 352), 'col_slice': (0, 192),   'operator': 'identity'},
)


def _make_background_pixel_map(background_value):
    return np.full((16, 178, 192), int(background_value), dtype=np.uint8)


def _raw_pixel_map_shape():
    return (4, 352, 384)


def _asic_pixel_map_shape():
    return (16, 178, 192)


def _normalize_raw_pixel_map(pixel_map_raw):
    arr = np.asarray(pixel_map_raw, dtype=np.uint8)
    if arr.shape == _raw_pixel_map_shape():
        return arr
    if arr.size == np.prod(_raw_pixel_map_shape()):
        return arr.reshape(_raw_pixel_map_shape())
    raise ValueError(
        f'user.pixel_map_raw expects shape {_raw_pixel_map_shape()}, got {arr.shape}'
    )


def _normalize_asic_pixel_map(pixel_map):
    arr = np.asarray(pixel_map, dtype=np.uint8)
    if arr.shape == _asic_pixel_map_shape():
        return arr
    if arr.size == np.prod(_asic_pixel_map_shape()):
        return arr.reshape(_asic_pixel_map_shape())
    raise ValueError(
        f'user.pixel_map expects shape {_asic_pixel_map_shape()}, got {arr.shape}'
    )


def _asic_pixel_map_to_raw_pixel_map(pixel_map):
    arr = _normalize_asic_pixel_map(pixel_map)
    out = np.zeros(_raw_pixel_map_shape(), dtype=np.uint8)

    for segment in range(4):
        for layout in RAW_MASK_ASIC_LAYOUT:
            r0, r1 = layout['row_slice']
            c0, c1 = layout['col_slice']
            operator = layout['operator']
            asic = 4 * segment + layout['slot']
            readable = np.asarray(arr[asic, :176, :], dtype=np.uint8)
            if operator == 'identity':
                out[segment, r0:r1, c0:c1] = readable
            elif operator == 'rot180':
                out[segment, r0:r1, c0:c1] = np.flipud(np.fliplr(readable))
            else:
                raise ValueError(f'unsupported raw pixel-map operator: {operator!r}')

    return out


def _config_entry_exists(cfg, dotted_key):
    node = cfg
    for key in dotted_key.split('.'):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _copy_entry_or_fallback_type(cfg, src_cfg, dotted_key, fallback_type_key=None):
    copy_config_entry(cfg, src_cfg, dotted_key)
    try:
        copy_config_entry(cfg[':types:'], src_cfg[':types:'], dotted_key)
    except KeyError:
        if fallback_type_key is None:
            raise
        copy_config_entry(cfg[':types:'], src_cfg[':types:'], fallback_type_key)


def _get_cfg_pixel_map_raw(cfg, fallback_cfg=None):
    candidates = [cfg]
    if fallback_cfg is not None and fallback_cfg is not cfg:
        candidates.append(fallback_cfg)

    for candidate in candidates:
        if not isinstance(candidate, dict) or 'user' not in candidate:
            continue
        user = candidate['user']
        if 'pixel_map_raw' in user:
            return _normalize_raw_pixel_map(user['pixel_map_raw'])
        if 'pixel_map' in user:
            return _asic_pixel_map_to_raw_pixel_map(user['pixel_map'])

    raise KeyError('user.pixel_map_raw or user.pixel_map is required')


def _get_cfg_trbit_by_asic(cfg, fallback_cfg=None):
    trbits = []
    for i in range(16):
        value = None
        for candidate in (cfg, fallback_cfg):
            if not isinstance(candidate, dict):
                continue
            try:
                value = candidate['expert']['EpixQuad'][f'Epix10kaSaci{i}']['trbit']
                break
            except KeyError:
                continue
        if value is None:
            raise KeyError(f'expert.EpixQuad.Epix10kaSaci{i}.trbit is required')
        trbits.append(int(value))
    return trbits


def _trbits_asic_to_panel_order(trbits_asic):
    trbits_asic = list(trbits_asic)
    if len(trbits_asic) != 16:
        raise ValueError(f'expected 16 ASIC trbits, got {len(trbits_asic)}')

    # Use the raw-panel ASIC placement already encoded in RAW_MASK_ASIC_LAYOUT.
    # Panel order here is [top-left, top-right, bottom-left, bottom-right].
    ordered_layout = sorted(
        RAW_MASK_ASIC_LAYOUT,
        key=lambda layout: (layout['row_slice'][0], layout['col_slice'][0]),
    )

    panel_trbits = []
    for segment in range(4):
        for layout in ordered_layout:
            asic = 4 * segment + layout['slot']
            panel_trbits.append(int(trbits_asic[asic]))

    return panel_trbits


def _analysis_config_from_cfg(cfg, fallback_cfg=None):
    pixel_map_raw = _get_cfg_pixel_map_raw(cfg, fallback_cfg=fallback_cfg)
    trbits_asic = _get_cfg_trbit_by_asic(cfg, fallback_cfg=fallback_cfg)
    trbits_panel = _trbits_asic_to_panel_order(trbits_asic)
    return pixel_map_raw, trbits_panel


def _segment_trbits_panel(trbits_panel, seg):
    start = 4 * seg
    stop = start + 4
    return list(trbits_panel[start:stop])


def _convert_raw_mask_to_direct_ops(mask, *, selected_value):
    """Converts a det.raw.raw-oriented mask into direct bank-addressed writes.

    Parameters
    ----------
    mask : np.ndarray
        Raw-view mask with shape (4,352,384). Nonzero pixels are treated as
        active and converted to direct writes.
    selected_value : int
        Pixel value to program for selected pixels.

    Returns
    -------
    ops : list[dict]
        Direct-write pixel ops in (asic, bank, row, col).
    summary : list[dict]
        Short per-ASIC summaries for logging.
    """
    if mask.shape != (4, 352, 384):
        raise ValueError(
            f'EPIXQUAD_DEBUG_MASK_NPY expects shape (4,352,384), got {mask.shape}'
        )

    active = np.asarray(mask) != 0
    ops = []
    summary = []
    for segment in range(4):
        seg_active = active[segment]
        for layout in RAW_MASK_ASIC_LAYOUT:
            r0, r1 = layout['row_slice']
            c0, c1 = layout['col_slice']
            sub = seg_active[r0:r1, c0:c1]
            coords = np.argwhere(sub)
            if coords.size == 0:
                continue

            asic = 4 * segment + layout['slot']
            operator = layout['operator']
            for raw_local_row, raw_local_col in coords:
                if operator == 'identity':
                    prog_row = int(raw_local_row)
                    prog_col = int(raw_local_col)
                elif operator == 'rot180':
                    prog_row = 175 - int(raw_local_row)
                    prog_col = 191 - int(raw_local_col)
                else:
                    raise ValueError(f'unsupported raw-mask operator: {operator!r}')

                bank = prog_col // 48
                bank_col = prog_col % 48
                ops.append({
                    'kind': 'pixel',
                    'asic': int(asic),
                    'bank': int(bank),
                    'row': int(prog_row),
                    'col': int(bank_col),
                    'value': int(selected_value),
                })

            summary.append({
                'asic': int(asic),
                'segment': int(segment),
                'operator': operator,
                'active_pixels': int(coords.shape[0]),
                'raw_box': ((int(r0), int(r1)), (int(c0), int(c1))),
            })
    return ops, summary


def _load_debug_mask_npy_override(user_cfg):
    """Optional env-gated debug override from a raw-view mask .npy file.

    Expected input mask orientation matches det.raw.raw(evt): shape (4,352,384).
    The converter applies the measured ASIC subregion/operator mapping and emits
    direct bank-addressed writes under the bank_rc_178x48 convention.
    """
    path = os.environ.get('EPIXQUAD_DEBUG_MASK_NPY')
    if not path:
        return None

    selected_value = int(os.environ.get('EPIXQUAD_DEBUG_MASK_SELECTED_VALUE', '8'))
    background_value = int(os.environ.get('EPIXQUAD_DEBUG_MASK_BACKGROUND_VALUE', '12'))
    trbit = int(os.environ.get('EPIXQUAD_DEBUG_MASK_TRBIT', '0'))

    mask = np.load(path)
    ops, summary = _convert_raw_mask_to_direct_ops(mask, selected_value=selected_value)
    stem = os.path.splitext(os.path.basename(path))[0]

    return {
        'override_kind': 'direct_write',
        'source_kind': 'mask_npy',
        'selection_source': 'env_mask_npy',
        'source_file': os.path.abspath(path),
        'direct_name': f'raw_mask_{stem}',
        'pattern_label': 'raw-mask direct write',
        'coordinate_mode': 'bank_rc_178x48',
        'pixel_map': _make_background_pixel_map(background_value),
        'trbit_by_asic': [trbit] * 16,
        'background_value_by_asic': [background_value] * 16,
        'selected_value': int(selected_value),
        'background_value': int(background_value),
        'direct_write_ops': ops,
        'direct_write_summary': ops[:6],
        'raw_mask_summary': summary,
        'direct_write_pixel_count': len(ops),
    }


def _format_direct_op_preview(op):
    if op['kind'] == 'bank_fill':
        row_range = op.get('row_range', (op['row_start'], op['row_stop']))
        col_range = op.get('col_range', (op['col_start'], op['col_stop']))
        return 'fill:a%d:b%d:r[%d,%d):c[%d,%d):v%d' % (
            op['asic'],
            op['bank'],
            row_range[0],
            row_range[1],
            col_range[0],
            col_range[1],
            op['value'],
        )
    return 'pixel:a%d:b%d:r%d:c%d:v%d' % (
        op['asic'],
        op['bank'],
        op['row'],
        op['col'],
        op['value'],
    )


def _apply_debug_pattern_override(cfg):
    """Optionally overrides config from debug test definitions.

    Two usage modes are supported.

    Standalone test-file mode:
      EPIXQUAD_DEBUG_TEST_FILE=/path/to/test.json
      EPIXQUAD_DEBUG_MARKER_GROUPS=group1[,group2,...]
      EPIXQUAD_DEBUG_GROUP_INDEX=<int>

      Use this when you want to run a single test JSON directly. If the selected
      test file contains multiple marker groups, you can select them either by
      name with EPIXQUAD_DEBUG_MARKER_GROUPS or by first-seen index with
      EPIXQUAD_DEBUG_GROUP_INDEX. If neither is set, the loader defaults to
      group 0.

    Sequence-pattern mode:
      EPIXQUAD_DEBUG_SEQUENCE_FILE=/path/to/sequence.json
      EPIXQUAD_DEBUG_PATTERN_INDEX=<int>

      Use this when a wrapper/scan driver iterates through a sequence file. Each
      sequence pattern selects one test file and usually one marker group for
      that run. EPIXQUAD_DEBUG_PATTERN_INDEX defaults to 0. The legacy
      EPIXQUAD_DEBUG_STEP_INDEX name is still accepted for compatibility.

    Optional in both modes:
      EPIXQUAD_DEBUG_PATTERN_OUTDIR=/path/to/save/materialized/patterns

    Direct register-write mode:
      EPIXQUAD_DEBUG_DIRECT_WRITE_FILE=/path/to/direct.json
      EPIXQUAD_DEBUG_PATTERN_INDEX=<int>

      Use this when you want to bypass the current logical pixel-map to
      bank/local-coordinate reconstruction and instead program explicit
      ASIC-local writes. In this mode the loader provides:
        - background value per ASIC
        - per-ASIC trbits
        - explicit write operations in (asic, bank, row, col)
      A background-only placeholder user.pixel_map_raw is still injected for
      metadata/debugging, but the actual hardware writes happen later in
      config_expert() from the direct op list.

    Raw-mask .npy mode:
      EPIXQUAD_DEBUG_MASK_NPY=/path/to/mask.npy
      EPIXQUAD_DEBUG_MASK_SELECTED_VALUE=8      # optional
      EPIXQUAD_DEBUG_MASK_BACKGROUND_VALUE=12   # optional
      EPIXQUAD_DEBUG_MASK_TRBIT=0               # optional

      Use this when you already have a raw-view mask with the same shape and
      orientation as det.raw.raw(evt), namely (4,352,384). The converter uses
      the measured ASIC/operator mapping and emits direct bank-addressed pixel
      writes under the bank_rc_178x48 convention. This bypasses the current
      logical pixel_map reconstruction path entirely.

    Wrapper/control-file mode:
      A client-side wrapper can update a shared JSON control file before each
      run. This is needed when the DAQ/config process is already running and
      environment-variable changes in the client will not propagate into that
      process. The control-file path defaults to the loader's built-in path
      unless overridden by EPIXQUAD_DEBUG_CONTROL_FILE in the DAQ process
      environment.

    When enabled, this function forces:
      cfg['user']['gain_mode'] = 5
      cfg['user']['pixel_map_raw'] = materialized raw-view array (4,352,384)
      cfg['expert']['EpixQuad']['Epix10kaSaci{i}']['trbit'] per loaded test

    If none of EPIXQUAD_DEBUG_TEST_FILE, EPIXQUAD_DEBUG_SEQUENCE_FILE,
    EPIXQUAD_DEBUG_DIRECT_WRITE_FILE, or EPIXQUAD_DEBUG_MASK_NPY is set, this
    function leaves cfg unchanged.
    """
    global debug_override

    materialized = load_debug_override(cfg.get('user'))
    if materialized is None:
        materialized = _load_debug_mask_npy_override(cfg.get('user'))
    debug_override = materialized
    if materialized is None:
        return False

    cfg.setdefault('user', {})
    cfg.setdefault('expert', {})
    cfg['expert'].setdefault('EpixQuad', {})

    cfg['user']['gain_mode'] = 5
    cfg['user']['pixel_map_raw'] = _get_cfg_pixel_map_raw(
        {'user': materialized}
    ).reshape(-1).tolist()
    for i, trbit in enumerate(materialized['trbit_by_asic']):
        cfg['expert']['EpixQuad'].setdefault(f'Epix10kaSaci{i}', {})
        cfg['expert']['EpixQuad'][f'Epix10kaSaci{i}']['trbit'] = int(trbit)

    log_parts = [
        f"loaded debug override kind={materialized.get('override_kind', 'unknown')}",
        f"source={materialized['source_kind']}",
        f"selection_source={materialized.get('selection_source', 'unknown')}",
        f"source_file={materialized['source_file']}",
    ]
    if materialized.get('override_kind') == 'pixel_map':
        marker_preview = ', '.join(
            '%s:a%d:r%d:c%d:v%d' % (
                m.get('label', '?'),
                m['asic'],
                m['row'],
                m['col'],
                m['value'],
            )
            for m in materialized.get('selected_markers', [])[:6]
        )
        log_parts.append(f"test={materialized['test_name']}")
        log_parts.append(f"groups={materialized.get('selected_groups', [])}")
        log_parts.append(f"active_pixels={materialized.get('active_pixel_count', 0)}")
        if marker_preview:
            log_parts.append(f"markers={marker_preview}")
    else:
        op_preview = ', '.join(
            _format_direct_op_preview(op)
            for op in materialized.get('direct_write_summary', [])[:6]
        )
        log_parts.append(f"direct={materialized.get('direct_name', 'unnamed_direct')}")
        log_parts.append(f"pattern_label={materialized.get('pattern_label', '')}")
        log_parts.append(f"coordinate_mode={materialized.get('coordinate_mode', 'bank_rc_178x48')}")
        log_parts.append(f"direct_pixels={materialized.get('direct_write_pixel_count', 0)}")
        log_parts.append('pixel_map_metadata=background_only_placeholder')
        if materialized.get('source_kind') == 'mask_npy':
            log_parts.append(
                f"mask_selected_value={materialized.get('selected_value', '?')}"
            )
            log_parts.append(
                f"mask_background_value={materialized.get('background_value', '?')}"
            )
        if op_preview:
            log_parts.append(f"ops={op_preview}")
    if 'sequence_name' in materialized:
        log_parts.append(f"sequence={materialized['sequence_name']}")
        log_parts.append(f"pattern_index={materialized['pattern_index']}")
        if 'pattern_label' in materialized:
            log_parts.append(f"pattern_label={materialized['pattern_label']}")
        if 'test_file' in materialized:
            log_parts.append(f"test_file={materialized['test_file']}")
    if 'control_file' in materialized:
        log_parts.append(f"control_file={materialized['control_file']}")
    logging.warning(' '.join(log_parts))
    return True


def _apply_direct_write_program(cbase, program, asics):
    """Programs explicit ASIC-local writes for debug-only bank/pixel probes.

    This bypasses the existing detector-view to bank/local-coordinate
    reconstruction logic and writes register coordinates directly. Two direct
    coordinate modes are supported:

      bank_rc_178x48
        - logical bank = hardware bank
        - logical row = hardware row
        - logical col = hardware bank-local col

      bank_rc_44x192
        - logical bank selects a 44-row band inside the ASIC
        - logical row is local within that 44-row band
        - logical col spans the full ASIC width and is expanded across the
          four hardware bank offsets
    """
    if not program or program.get('override_kind') != 'direct_write':
        return False

    coordinate_mode = program.get('coordinate_mode', 'bank_rc_178x48')

    def iter_hw_targets(op):
        if coordinate_mode == 'bank_rc_178x48':
            if op['kind'] == 'pixel':
                yield (int(op['row']), int(op['bank']), int(op['col']), int(op['value']))
                return
            for row in range(int(op['row_start']), int(op['row_stop'])):
                for col in range(int(op['col_start']), int(op['col_stop'])):
                    yield (row, int(op['bank']), col, int(op['value']))
            return

        if coordinate_mode == 'bank_rc_44x192':
            row_base = int(op['bank']) * 44
            if op['kind'] == 'pixel':
                logical_col = int(op['col'])
                yield (
                    row_base + int(op['row']),
                    logical_col // 48,
                    logical_col % 48,
                    int(op['value']),
                )
                return
            for row in range(int(op['row_start']), int(op['row_stop'])):
                hw_row = row_base + row
                for logical_col in range(int(op['col_start']), int(op['col_stop'])):
                    yield (hw_row, logical_col // 48, logical_col % 48, int(op['value']))
            return

        raise ValueError(f'unsupported direct coordinate_mode: {coordinate_mode!r}')

    for i in asics:
        saci = cbase.Epix10kaSaci[i]
        saci.PrepareMultiConfig.set(0)

    background_value_by_asic = program['background_value_by_asic']
    for i in asics:
        saci = cbase.Epix10kaSaci[i]
        saci.WriteMatrixData.set(int(background_value_by_asic[i]))

    direct_write_t0 = time.perf_counter()
    changed_pixels = 0
    for op in program['direct_write_ops']:
        saci = cbase.Epix10kaSaci[int(op['asic'])]
        for hw_row, hw_bank, hw_col, hw_value in iter_hw_targets(op):
            bank_offset = BANK_OFFSETS[int(hw_bank)]
            saci.RowCounter.set(int(hw_row))
            saci.ColCounter.set(bank_offset | int(hw_col))
            saci.WritePixelData.set(int(hw_value))
            changed_pixels += 1

    logging.info(
        'Direct debug write complete in %.3f s (%d explicit pixel updates, coordinate_mode=%s)',
        time.perf_counter() - direct_write_t0,
        changed_pixels,
        coordinate_mode,
    )
    return True


def _apply_raw_pixel_map_program(cbase, pixel_map_raw, asics):
    if pixel_map_raw.shape != (4, 352, 384):
        raise ValueError(
            f'user.pixel_map_raw expects shape (4,352,384), got {pixel_map_raw.shape}'
        )

    for i in asics:
        saci = cbase.Epix10kaSaci[i]
        saci.PrepareMultiConfig.set(0)

    matrix_cfg_t0 = time.perf_counter()
    changed_pixels = 0

    for segment in range(4):
        seg = pixel_map_raw[segment]
        for layout in RAW_MASK_ASIC_LAYOUT:
            r0, r1 = layout['row_slice']
            c0, c1 = layout['col_slice']
            operator = layout['operator']
            asic = 4 * segment + layout['slot']
            saci = cbase.Epix10kaSaci[asic]

            asic_raw = np.asarray(seg[r0:r1, c0:c1], dtype=np.uint8)
            common_value = mode(asic_raw)
            saci.WriteMatrixData.set(int(common_value))

            raw_rows, raw_cols = asic_raw.shape
            for raw_row in range(raw_rows):
                for raw_col in range(raw_cols):
                    pixel_value = int(asic_raw[raw_row, raw_col])
                    if pixel_value == common_value:
                        continue

                    if operator == 'identity':
                        prog_row = raw_row
                        prog_col = raw_col
                    elif operator == 'rot180':
                        prog_row = (raw_rows - 1) - raw_row
                        prog_col = (raw_cols - 1) - raw_col
                    else:
                        raise ValueError(f'unsupported raw pixel-map operator: {operator!r}')

                    bank = prog_col // 48
                    bank_col = prog_col % 48
                    saci.RowCounter.set(int(prog_row))
                    saci.ColCounter.set(BANK_OFFSETS[int(bank)] | int(bank_col))
                    saci.WritePixelData.set(pixel_value)
                    changed_pixels += 1

    logging.info(
        'Raw pixel_map write complete in %.3f s (%d pixel updates)',
        time.perf_counter() - matrix_cfg_t0,
        changed_pixels,
    )
    return True

def get_trigger_buffers():
    """
    Returns the Run/DAQ trigger buffer indices for the current PGP lane.

    Firmware mapping:
        TriggerEventBuffer[lane]     → DAQ trigger (XPM, beam-synced, ~100 Hz)
        TriggerEventBuffer[lane + 4] → Run trigger (EVR event-code 6, 1080 Hz)

    Returns
    -------
    run_buf : int
        Index of the Run trigger buffer.
    daq_buf : int
        Index of the DAQ trigger buffer.
    """
    global lane
    return lane + 4, lane  # (run_buf, daq_buf)

def calc_daq_trigger_delay(base, rawStart_ns, group):
    """
    Compute DAQ TriggerEventBuffer trigger delay based on partitionDelay.
    """
    pbase = base['pci']
    clk_period = base['clk_period']
    msg_period = base['msg_period']
    partitionDelay = getattr(
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,
        f'PartitionDelay[{group}]'
    ).get()
    triggerDelay = int(rawStart_ns / clk_period - partitionDelay * msg_period + 9)
    if triggerDelay < 0:
        logging.warning(f"DAQ triggerDelay computed negative ({triggerDelay}), clamping to 0")
        triggerDelay = 0
    return triggerDelay


def calc_run_trigger_delay(base, rawStart_ns):
    """
    Compute Run TriggerEventBuffer trigger delay based on EVR delay line.
    """
    clk_period = base['clk_period']
    triggerDelay = int(rawStart_ns / clk_period)
    return triggerDelay

def mode(a):
    uniqueValues = np.unique(a).tolist()
    uniqueCounts = [len(np.nonzero(a == uv)[0])
                    for uv in uniqueValues]

    modeIdx = uniqueCounts.index(max(uniqueCounts))
    return uniqueValues[modeIdx]

def dumpvars(prefix,c):
    print(prefix)
    for key,val in c.nodes.items():
        name = prefix+'.'+key
        dumpvars(name,val)

def retry(cmd,val):
    itry=0
    while(True):
        try:
            cmd(val)
        except Exception as e:
            logging.warning(f'Try {itry} of {cmd}({val}) failed.')
            if itry < 3:
                itry+=1
                continue
            else:
                raise e
        break

def _hydrate_map_mode_partial_config(cfg):
    if ':types:' not in cfg:
        cfg[':types:'] = {}

    if 'user' not in cfg or 'gain_mode' not in cfg['user']:
        return

    if cfg['user']['gain_mode'] != 5 or 'pixel_map_raw' in cfg['user']:
        return

    if ocfg is None or 'user' not in ocfg:
        raise KeyError('full configuration is required when user.gain_mode == 5')

    if _config_entry_exists(ocfg, 'user.pixel_map_raw'):
        _copy_entry_or_fallback_type(cfg, ocfg, 'user.pixel_map_raw')
    elif _config_entry_exists(ocfg, 'user.pixel_map'):
        cfg.setdefault('user', {})
        cfg['user']['pixel_map_raw'] = _asic_pixel_map_to_raw_pixel_map(
            ocfg['user']['pixel_map']
        ).reshape(-1).tolist()
        _copy_entry_or_fallback_type(
            cfg,
            ocfg,
            'user.pixel_map_raw',
            fallback_type_key='user.pixel_map',
        )
    else:
        raise KeyError('user.pixel_map_raw or user.pixel_map is required when user.gain_mode == 5')

    for i in range(16):
        key = f'expert.EpixQuad.Epix10kaSaci{i}.trbit'
        copy_config_entry(cfg, ocfg, key)
        copy_config_entry(cfg[':types:'], ocfg[':types:'], key)

#
#  Apply the configuration dictionary to the rogue registers
#
def apply_dict(pathbase,base,cfg):
    rogue_translate = {}
    rogue_translate['TriggerEventBuffer'] = f'TriggerEventBuffer[{lane}]'
    for i in range(16):
        rogue_translate[f'Epix10kaSaci{i}'] = f'Epix10kaSaci[{i}]'
    for i in range(3):
        rogue_translate[f'DbgOutSel{i}'] = f'DbgOutSel[{i}]'

    depth = 0
    my_queue  =  deque([[pathbase,depth,base,cfg]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        if(dict is type(configdb_node)):
            for i in configdb_node:
                k = rogue_translate[i] if i in rogue_translate else i
                try:
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[k],configdb_node[i]])
                except KeyError:
                    logging.warning('Lookup failed for node [{:}] in path [{:}]'.format(i,path))

        #  Apply
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path != pathbase ):
#            if False:
            if (('Saci' in path and 'PixelDummy' in path) or
                ('Saci3' in path and 'CompEn' in path) or
                ('Saci3' in path and 'Preamp' in path) or
                ('Saci3' in path and 'MonostPulser' in path) or
                ('Saci3' in path and 'PulserDac' in path) or
                ('PseudoScopeCore' in path)):  #  Writes fail -- fix me!
                logging.info(f'NOT setting {path} to {configdb_node}')
            else:
                logging.info(f'Setting {path} to {configdb_node}')
                retry(rogue_node.set,configdb_node)

#
#  Construct an asic pixel mask with square spacing
#
def pixel_mask_square(value0,value1,spacing,position):
    ny,nx=352,384;
    if position>=spacing**2:
        logging.error('position out of range')
        position=0;
    out=np.zeros((ny,nx),dtype=np.int32)+value0
    position_x=position%spacing; position_y=position//spacing
    out[position_y::spacing,position_x::spacing]=value1
    return out

#
#  Initialize the rogue accessor
#
def epixquad_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M", verbose=0):
    global base
    global pv
    global lane
    if verbose and False:  # pyrogue prevents us from using DEBUG here
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.debug('epixquad_init')

    base = {}

    #  Configure the PCIe card first (timing, datavctap)
    if True:
        pbase = lcls2_pgp_pcie_apps.DevRoot(dev           =dev,
                                            enLclsI       =False,
                                            enLclsII      =True,
                                            yamlFileLclsI =None,
                                            yamlFileLclsII=None,
                                            startupMode   =True,
                                            standAloneMode=xpmpv is not None,
                                            pgp4          =True,
                                            dataVc        =0,
                                            pollEn        =False,
                                            initRead      =False)
        #dumpvars('pbase',pbase)

        pbase.__enter__()

        # Set the XPM pause threshold on the DDR buffer
        appLane = pbase.find(typ=shared.AppLane)
        for devPtr in appLane:
            devPtr.XpmPauseThresh.set(0x20)
            devPtr.EventBuilder.Timeout.set(int(156.25e6/1080))

        #  Disable flow control on the PGP lane at the PCIe end
#        getattr(pbase.DevPcie.Hsio,f'PgpMon[{lane}]').Ctrl.FlowControlDisable.set(1)

        # Open a new thread here
        if xpmpv is not None:
            pv = PVCtrls(xpmpv,pbase.DevPcie.Hsio.TimingRx.XpmMiniWrapper)
            pv.start()
        else:
            time.sleep(0.1)
        base['pci'] = pbase

    #  Connect to the camera
    cbase = ePixQuad.Top(dev=dev,hwType='datadev',lane=lane,pollEn=False,
                         enVcMask=0x2,enWriter=False,enPrbs=False)
    cbase.__enter__()
    base['cam'] = cbase

    firmwareVersion = cbase.AxiVersion.FpgaVersion.get()
    buildStamp = cbase.AxiVersion.BuildStamp.get()
    gitHash = cbase.AxiVersion.GitHash.get()
    print(f'firmwareVersion [{firmwareVersion:x}]')
    print(f'buildStamp      [{buildStamp}]')
    print(f'gitHash         [{gitHash:x}]')

    if DEBUG_ADC_TRAIN_WRITE:
        print("[DEBUG-ADC] Reading ADC calibration constants from PROM...")
        cbase.CypressS25Fl.readCmd(0x3000000)
        adc_data = cbase.CypressS25Fl.getDataReg()
        print(f"[DEBUG-ADC] ADC calibration data ({len(adc_data)} bytes):", adc_data)
        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = "/tmp"
        os.makedirs(outdir, exist_ok=True)

        # Save to .npy
        npy_file = os.path.join(outdir, f"adc_training_{ts}.npy")
        np.save(npy_file, np.array(adc_data))
        print(f"[DEBUG-ADC] Saved ADC training data to {npy_file}")

    logging.info('epixquad_unconfig')
    epixquad_unconfig(base)

    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ModeSelEn.set(1)
    if timebase=="119M":
        logging.info('Using timebase 119M')
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(0)
    else:
        logging.info('Using timebase 186M')
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(1)
    #  To get the timing feedback link working
    pbase.DevPcie.Hsio.TimingRx.TimingPhyMonitor.TxPhyPllReset()
    time.sleep(1)
    #  Reset rx with the new reference
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.C_RxReset()
    time.sleep(2)
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.RxDown.set(0)

    #
    # Configure Run/DAQ trigger buffers (new dual-buffer scheme)
    #
    run_buf, daq_buf = get_trigger_buffers()
    trigman = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager

    logging.info(f"Configuring Run/DAQ triggers for lane {lane}: run_buf={run_buf}, daq_buf={daq_buf}")

    # --- Run trigger: EVR event code 6 (~1080 Hz)

    trigman.TriggerEventBuffer[run_buf].TriggerSource.set(1)  # EVR
    trigman.EvrV2CoreTriggers.EvrV2ChannelReg[run_buf].EnableReg.set(1)
    trigman.EvrV2CoreTriggers.EvrV2ChannelReg[run_buf].RateType.set(2)  # EventCode mode/ControlWord
    trigman.EvrV2CoreTriggers.EvrV2ChannelReg[run_buf].RateSel.set(6)   # EventCode 6 = 1080 Hz
    trigman.EvrV2CoreTriggers.EvrV2ChannelReg[run_buf].DestType.set(2)  # All
    trigman.EvrV2CoreTriggers.EvrV2TriggerReg[run_buf].EnableTrig.set(1)
    trigman.EvrV2CoreTriggers.EvrV2TriggerReg[run_buf].Source.set(run_buf)
    trigman.EvrV2CoreTriggers.EvrV2TriggerReg[run_buf].Polarity.set(1)  # Rising
    trigman.EvrV2CoreTriggers.EvrV2TriggerReg[run_buf].Width.set(1)
    trigman.TriggerEventBuffer[run_buf].MasterEnable.set(1)

    # --- DAQ trigger: XPM, partition-based (~100 Hz)
    trigman.TriggerEventBuffer[daq_buf].TriggerSource.set(0)  # XPM
    # Partition (readout group) will be configured later in epixquad_config()
    # Delay tuned by user.start_ns via user_to_expert()
    logging.info("Run/DAQ trigger buffers configured")

    # We stay in expternal trigger mode througout
    epixquad_external_trigger(base)
    return base

#
#  Set the PGP lane
#
def epixquad_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

#
#  Set the local timing ID and fetch the remote timing ID
#
def epixquad_connectionInfo(base, alloc_json_str):

    if 'pci' in base:
        pbase = base['pci']
        rxId = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        logging.info('RxId {:x}'.format(rxId))
        txId = timTxId('epixquad')
        logging.info('TxId {:x}'.format(txId))
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    else:
        rxId = 0xffffffff


    epixquadid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixquadid

    return d

#
#  Translate the 'user' components of the cfg dictionary into 'expert' settings
#  The cfg dictionary may be partial (scanning), so the ocfg dictionary is
#  reference for the full set.
#
def user_to_expert(base, cfg, full=False):
    global ocfg
    global group
    global lane
    global debug_override

    _apply_debug_pattern_override(cfg)

    pbase = base['pci']

    d = {}
    hasUser = 'user' in cfg
    if hasUser and 'start_ns' in cfg['user']:
        rawStart = cfg['user']['start_ns']
        run_buf, daq_buf = get_trigger_buffers()

        # --- DAQ trigger delay (XPM)
        daq_triggerDelay = calc_daq_trigger_delay(base, rawStart, group)
        #d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[{daq_buf}].TriggerDelay'] = daq_triggerDelay

        # --- Run trigger delay (EVR event-code)
        run_delay = calc_run_trigger_delay(base, rawStart)
        #d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[{run_buf}].Delay'] = run_delay

    # Previously, gate_ns (user-requested acquisition window in nanoseconds)
    # was used to compute AsicAcqWidth. This option has been removed.
    # The default acquisition window is fixed to 100 ms (100_000_000 ns).
    # The corresponding AsicAcqWidth value should be (100_000 / 6.4) ≈ 15625,
    # where 6.4 ns is the sysclk period for ePixQuad 1kfps.

    ASIC_SYSCLK_NS = 6.4  # nanoseconds per sysclk tick for this camera
    expected_trigger_width = int(100_000 / ASIC_SYSCLK_NS)  # ~15625 ticks

    # Warn if the deprecated user field still exists
    if hasUser and 'gate_ns' in cfg['user']:
        logging.warning(
            "User parameter 'gate_ns' has been removed. "
            "The acquisition window is fixed to 100 ms. "
            "Ignoring user-specified value (%s ns).",
            cfg['user']['gate_ns']
        )

    # Read AsicAcqWidth directly from firmware
    try:
        cbase = base['cam']
        current_val = cbase.AcqCore.AsicAcqWidth.get()
        logging.info(
            "Firmware AsicAcqWidth = %d (expected %d for 100 ms window, sysclk = %.1f ns)",
            current_val, expected_trigger_width, ASIC_SYSCLK_NS
        )
    except Exception as e:
        logging.warning(
            "Could not read AsicAcqWidth from firmware: %s. "
            "Expected ≈ %d for 100 ms window (sysclk = %.1f ns).",
            e, expected_trigger_width, ASIC_SYSCLK_NS
        )


    if full:
        d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.Partition']=group

    pixel_map_changed = False
    a = None
    if (hasUser and ('gain_mode' in cfg['user'] or
                     'pixel_map' in cfg['user'] or
                     'pixel_map_raw' in cfg['user'])):
        gain_mode = cfg['user'].get('gain_mode', ocfg['user']['gain_mode'])
        if gain_mode==5:
            if 'pixel_map_raw' in cfg['user']:
                a_raw = _normalize_raw_pixel_map(cfg['user']['pixel_map_raw'])
                d['user.pixel_map_raw'] = a_raw.reshape(-1).tolist()
                logging.debug('pixel_map_raw len {}'.format(len(d['user.pixel_map_raw'])))
                logging.info(
                    'Prepared user.pixel_map_raw for config update: shape=%s nonzero_pixels=%d',
                    a_raw.shape,
                    int(np.count_nonzero(a_raw)),
                )
                pixel_map_changed = True
            elif 'pixel_map' in cfg['user']:
                a_raw = _asic_pixel_map_to_raw_pixel_map(cfg['user']['pixel_map'])
                d['user.pixel_map_raw'] = a_raw.reshape(-1).tolist()
                logging.debug('pixel_map_raw len {}'.format(len(d['user.pixel_map_raw'])))
                pixel_map_changed = True
            else:
                raise KeyError('user.pixel_map_raw or user.pixel_map is required when user.gain_mode == 5')
        else:
            mapv  = (0xc,0xc,0x8,0x0,0x0)[gain_mode] # H/M/L/AHL/AML
            trbit = (0x1,0x0,0x0,0x1,0x0)[gain_mode]
            base_raw = _get_cfg_pixel_map_raw(ocfg)
            a_raw = (np.asarray(base_raw, dtype=np.uint8) & 0x3) | mapv

            for i in range(16):
                d[f'expert.EpixQuad.Epix10kaSaci{i}.trbit'] = trbit
            logging.debug('pixel_map_raw len {}'.format(a_raw.size))
            d['user.pixel_map_raw'] = a_raw.reshape(-1).tolist()
            pixel_map_changed = True

    update_config_entry(cfg, ocfg, d)

    return pixel_map_changed

#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writePixelMap=True):

    # Turn off the trigger
    epixquad_disable_runtrigger(base)

    # overwrite the low-level configuration parameters with calculations from the user configuration
    pbase = base['pci']
    if ('expert' in cfg and 'DevPcie' in cfg['expert']):
        apply_dict('pbase.DevPcie',pbase.DevPcie,cfg['expert']['DevPcie'])

    cbase = base['cam']

    #  Make list of enabled ASICs
    asics = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    #  Important that Asic IsEn is True while configuring and false when running
    for i in asics:
        logging.debug(f'Enabling ASIC {i}')
        saci = cbase.Epix10kaSaci[i]
        retry(saci.enable.set,True)
        retry(saci.IsEn.set,True)

    if ('expert' in cfg and 'EpixQuad' in cfg['expert']):
        epixQuad = cfg['expert']['EpixQuad'].copy()
        #  Add write protection word to upper range
        if 'AcqCore' in epixQuad and 'AsicRoClkT' in epixQuad['AcqCore']:
            epixQuad['AcqCore']['AsicRoClkT'] |= 0xaaaa0000
        if 'AcqCore' in epixQuad and 'AsicRoClkHalfT' in epixQuad['AcqCore']:
            epixQuad['AcqCore']['AsicRoClkHalfT'] |= 0xaaaa0000
        if 'RdoutCore' in epixQuad and 'AdcPipelineDelay' in epixQuad['RdoutCore']:
            epixQuad['RdoutCore']['AdcPipelineDelay'] |= 0xaaaa0000
        apply_dict('cbase',cbase,epixQuad)

    if writePixelMap:
        if debug_override is not None and debug_override.get('override_kind') == 'direct_write':
            _apply_direct_write_program(cbase, debug_override, asics)
        elif 'user' in cfg and 'pixel_map_raw' in cfg['user']:
            pixelConfigMapRaw = _normalize_raw_pixel_map(cfg['user']['pixel_map_raw'])
            logging.info(
                'Using user.pixel_map_raw write path: shape=%s nonzero_pixels=%d unique_values=%s',
                pixelConfigMapRaw.shape,
                int(np.count_nonzero(pixelConfigMapRaw)),
                np.unique(pixelConfigMapRaw).tolist(),
            )
            _apply_raw_pixel_map_program(cbase, pixelConfigMapRaw, asics)
        elif 'user' in cfg and 'pixel_map' in cfg['user']:
            #  Write the pixel gain maps
            #  Would like to send a 3d array
            a = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
            pixelConfigMap = np.reshape(a,(16,178,192))

            # ***CAUTION ONLY FOR DEBUGGING *** Enable here to test pixel by pixel write
            if DEBUG_RANDOM_PIXEL_MAP:
                shape = (16, 178, 192)
                pixelConfigMap = np.random.choice([8, 12], size=shape, p=[0.5, 0.5]).astype(np.uint8)

            matrix_cfg_t0 = time.perf_counter()
            if USE_ACCELERATED_MATRIX_WRITE:
                #
                #  Accelerated matrix configuration (~2 seconds)
                #
                #  Found that gain_mode is mapping to [M/M/L/M/M]
                #    Like trbit is always zero (Saci was disabled!)
                #
                accel_t0 = time.perf_counter()
                core = cbase.SaciConfigCore
                core.enable.set(True)
                core.SetAsicsMatrix(json.dumps(pixelConfigMap.tolist()))
                core.enable.set(False)
                print(f'SetAsicsMatrix accelerated write took {time.perf_counter() - accel_t0:.3f} s')
                if DEBUG_PIXEL_MASK_SAVED:
                    saci = cbase.Epix10kaSaci[0].GetPixelBitmap("/tmp/pixel_mask.csv")
                    print(f"[DEBUG-FIXEDLOW] Wrote PixelBitmap for Asic0")


            else:
                #
                #  Pixel by pixel matrix configuration (up to 15 minutes)
                #
                #  Found that gain_mode is mapping to [H/M/M/H/M]
                #    Like pixelmap is always 0xc
                #
                for i in asics:
                    saci = cbase.Epix10kaSaci[i]
                    saci.PrepareMultiConfig.set(0)

                #  Set the whole ASIC to its most common value
                masic = {}
                for i in asics:
                    masic[i] = mode(pixelConfigMap[i])
                    saci = cbase.Epix10kaSaci[i]
                    saci.WriteMatrixData.set(masic[i])  # 0x4000 v 0x84000

                #  Now fix any pixels not at the common value
                changed_pixels = 0
                per_pixel_t0 = time.perf_counter()
                for i in asics:
                    saci = cbase.Epix10kaSaci[i]
                    nrows = pixelConfigMap.shape[1]
                    ncols = pixelConfigMap.shape[2]

                    writeView = pixelConfigMap[:, :nrows, :ncols]

                    for row in range(nrows):
                        for col in range(ncols):
                            if pixelConfigMap[i,row,col]!=masic[i]:
                                changed_pixels += 1
                                if row >= (nrows>>1):
                                    mrow = row - (nrows>>1)
                                    if col < (ncols>>1):
                                        offset = 3
                                        mcol = col
                                    else:
                                        offset = 0
                                        mcol = col - (ncols>>1)
                                else:
                                    mrow = (nrows>>1)-1 - row
                                    if col < (ncols>>1):
                                        offset = 2
                                        mcol = (ncols>>1)-1 - col
                                    else:
                                        offset = 1
                                        mcol = (ncols-1) - col
                                bank = int((mcol % (48<<2)) / 48)
                                bankOffset = BANK_OFFSETS[bank]
                                saci.RowCounter.set(row)
                                saci.ColCounter.set(bankOffset | (mcol%48))
                                saci.WritePixelData.set(int(pixelConfigMap[i,row,col]))
                logging.info('SetAsicsMatrix per-pixel write took %.3f s (%d pixel updates)',
                             time.perf_counter() - per_pixel_t0, changed_pixels)

                if DEBUG_PIXEL_MASK_SAVED:
                    saci = cbase.Epix10kaSaci[0].GetPixelBitmap("/tmp/pixel_mask.csv")
                    print(f"[DEBUG-FIXEDLOW] Wrote PixelBitmap for Asic0")

            logging.info(f'SetAsicsMatrix complete in {time.perf_counter() - matrix_cfg_t0:.3f} s')
        else:
            logging.info('writePixelMap but no new map')
            logging.debug(cfg)

    #  Important that Asic IsEn is True while configuring and false when running
    for i in asics:
        saci = cbase.Epix10kaSaci[i]
        retry(saci.IsEn.set,False)
        retry(saci.enable.set,False)

    # Turn back on Run Trigger
    epixquad_enable_runtrigger(base)


    logging.debug('config_expert complete')


def reset_counters(base):
    base['pci'].DevPcie.Hsio.TimingRx.TimingFrameRx.countReset()

    _, daq_buf = get_trigger_buffers()
    base['pci'].DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[daq_buf].countReset()

    base['cam'].RdoutStreamMonitoring.countReset()


def startRun(pbase):
    """
    Start DAQ acquisition for the detector's application lane.

    Arms the EventBuilder and enables only the DAQ trigger buffer
    (lane*2 + 1). The Run trigger (lane*2) remains active.
    """
    logging.info('StartRun() executed')

    run_buf, daq_buf = get_trigger_buffers()
    trig_mgr = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager
    eventBuilder = [getattr(pbase.DevPcie.Application, f'AppLane[{lane}]').EventBuilder]

    pbase.CountReset()

    for devPtr in eventBuilder:
        devPtr.Blowoff.set(False)
        devPtr.SoftRst()

    trig_mgr.TriggerEventBuffer[daq_buf].MasterEnable.set(True)
    logging.info(f"Enabled DAQ Trigger buffer {daq_buf} (Run buffer {run_buf} remains active)")

    pbase.RunState.set(True)


def stopRun(pbase):
    """
    Stop DAQ acquisition for the detector's application lane.

    Disables only the DAQ trigger buffer (lane*2 + 1) while leaving
    the Run trigger (lane*2) active to keep the detector clocked.
    """
    logging.info('StopRun() executed')

    run_buf, daq_buf = get_trigger_buffers()
    trig_mgr = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager
    eventBuilder = [getattr(pbase.DevPcie.Application, f'AppLane[{lane}]').EventBuilder]

    try:
        trig_mgr.TriggerEventBuffer[daq_buf].MasterEnable.set(False)
        logging.info(f"Disabled DAQ Trigger buffer {daq_buf} (Run buffer {run_buf} remains active)")
    except Exception as e:
        logging.warning(f"Failed to disable DAQ Trigger buffer {daq_buf}: {e}")

    for devPtr in eventBuilder:
        devPtr.Blowoff.set(True)

    pbase.RunState.set(False)


#
#  Called on Configure
#
def epixquad_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global ocfg
    global group
    global segids

    group = rog

    _checkADCs()

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    #  Translate user settings to the expert fields
    user_to_expert(base, cfg, full=True)

    #  Apply the expert settings to the device
    config_expert(base, cfg)

    pbase = base['pci']

    run_buf, daq_buf = get_trigger_buffers()

    #  Force write Run/DAQ Trigger Delay here until configdb is fixed
    hasUser = 'user' in cfg
    if hasUser and 'start_ns' in cfg['user']:
        rawStart = cfg['user']['start_ns']
        # --- DAQ trigger delay (XPM)
        daq_triggerDelay = calc_daq_trigger_delay(base, rawStart, group)
        print(f"DAQ set TriggerEventBuffer[{daq_buf}].TriggerDelay = {daq_triggerDelay}")
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[daq_buf].TriggerDelay.set(daq_triggerDelay)

        # --- Run trigger delay (EVR event-code)
        run_delay = calc_run_trigger_delay(base, rawStart)
        print(f"Run set EvrV2TriggerReg[{run_buf}].Delay={run_delay}")
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[run_buf].Delay.set(run_delay)

    #
    # Configure DAQ trigger partition (XPM)
    #
    trigman = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager

    logging.info(f"Setting DAQ trigger buffer {daq_buf} Partition to group {group}")
    trigman.TriggerEventBuffer[daq_buf].Partition.set(group)

    #pbase.StartRun()
    startRun(pbase)

    #  Add some counter resets here
    reset_counters(base)

    #  Capture the firmware version to persist in the xtc
    cbase = base['cam']
    firmwareVersion = cbase.AxiVersion.FpgaVersion.get()

    #  *HOTFIX* From Julian's YML
    #  We need to set this only for epixquad 1080 because this value is set to 0x2 in the firmware.
    #  /cds/home/j/jumdz/epix-quad/software/yml/ued/epixQuad_ASICs_allAsics_UED_1080Hz_settings.yml
    #  RdoutCore.AdcPipelineDelay is set in the configdb (value=61, or 0xaaaa003d)
    cbase.AcqCore.AsicRoClkT.set(int(0xaaaa0003))


    ocfg = cfg

    #
    #  Create the segment configurations from parameters required for analysis
    #
    topname = cfg['detName:RO'].split('_')

    scfg = {}
    segids = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]

    pixelConfigMap, trbit = _analysis_config_from_cfg(cfg, fallback_cfg=ocfg)

    for seg in range(4):
        #  Construct the ID
        carrierId = [ cbase.SystemRegs.CarrierIdLow [seg].get(),
                      cbase.SystemRegs.CarrierIdHigh[seg].get() ]
        digitalId = [ 0, 0 ]
        analogId  = [ 0, 0 ]
        id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                          carrierId[0], carrierId[1],
                                                          digitalId[0], digitalId[1],
                                                          analogId [0], analogId [1])
        segids[seg] = id
        top = cdict()
        top.setAlg('config', [2,0,0])
        top.setInfo(detType='epix10ka', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
        top.set('asicPixelConfig', pixelConfigMap[seg].tolist(), 'UINT8')
        top.set('trbit'          , _segment_trbits_panel(trbit, seg), 'UINT8')
        scfg[seg+1] = top.typed_json()

    result = []
    for i in seglist:
        logging.debug('json seg {}  detname {}'.format(i, scfg[i]['detName:RO']))
        result.append( json.dumps(scfg[i]) )

    return result

def epixquad_unconfig(base):
    pbase = base['pci']
    #pbase.StopRun()
    stopRun(pbase)
    return base

#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixquad_scan_keys(update):
    logging.debug('epixquad_scan_keys')
    global ocfg
    global base
    global segids

    cfg = {}
    copy_reconfig_keys(cfg,ocfg,json.loads(update))
    _hydrate_map_mode_partial_config(cfg)
    # Apply to expert
    pixelMapChanged = user_to_expert(base,cfg,full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]

    if pixelMapChanged:
        pixelConfigMap, trbit = _analysis_config_from_cfg(cfg, fallback_cfg=ocfg)

        cbase = base['cam']
        for seg in range(4):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epix10ka', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigMap[seg].tolist(), 'UINT8')
            top.set('trbit'          , _segment_trbits_panel(trbit, seg), 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixquad_update(update):
    logging.debug('epixquad_update')
    global ocfg
    global base
    # extract updates
    cfg = {}
    update_config_entry(cfg,ocfg,json.loads(update))
    _hydrate_map_mode_partial_config(cfg)
    #  Apply to expert
    writePixelMap = user_to_expert(base,cfg,full=False)
    #  Apply config
    config_expert(base, cfg, writePixelMap)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]

    if writePixelMap:
        pixelConfigMap, trbit = _analysis_config_from_cfg(cfg, fallback_cfg=ocfg)

        cbase = base['cam']
        for seg in range(4):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epix10ka', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigMap[seg].tolist(), 'UINT8')
            top.set('trbit'          , _segment_trbits_panel(trbit, seg), 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

    logging.debug('update complete')

    return result

def epixquad_enable_runtrigger(base):
    pbase = base['pci']
    run_buf, daq_buf = get_trigger_buffers()
    trigman = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager
    trigman.TriggerEventBuffer[run_buf].MasterEnable.set(1)
    print(f'[DEBUG-RUNTRIG] Enable RunTrigger')


def epixquad_disable_runtrigger(base):
    pbase = base['pci']
    run_buf, daq_buf = get_trigger_buffers()
    trigman = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager
    trigman.TriggerEventBuffer[run_buf].MasterEnable.set(0)
    print(f'[DEBUG-RUNTRIG] Disable RunTrigger')

#
#  Check that ADC startup has completed successfully
#
def _checkADCs():

    epixquad_disable_runtrigger(base)

    cbase = base['cam']
    tmo = 0
    while True:
        time.sleep(0.001)
        if cbase.SystemRegs.AdcTestFailed.get()==1:
            logging.warning('Adc Test Failed - restarting!')
            cbase.SystemRegs.AdcReqStart.set(1)
            time.sleep(1.e-6)
            cbase.SystemRegs.AdcReqStart.set(0)
        else:
            tmo += 1
            if tmo > 1000:
                logging.warning('Adc Test Timedout')
                return 1
        if cbase.SystemRegs.AdcTestDone.get()==1:
            break
    logging.debug(f'Adc Test Done after {tmo} cycles')

    epixquad_enable_runtrigger(base)

    return 0

def _resetSequenceCount():
    cbase = base['cam']
    cbase.AcqCore.AcqCountReset.set(1)
    cbase.RdoutCore.SeqCountReset.set(1)
    time.sleep(1.e6)
    cbase.AcqCore.AcqCountReset.set(0)
    cbase.RdoutCore.SeqCountReset.set(0)

def epixquad_external_trigger(base):
    cbase = base['cam']
    #  Switch to external triggering
    cbase.SystemRegs.AutoTrigEn.set(0)
    cbase.SystemRegs.TrigSrcSel.set(0)
    cbase.SystemRegs.TrigEn.set(1)
    #  Enable frame readout
    cbase.RdoutCore.RdoutEn.set(1)

def epixquad_internal_trigger(base):
    cbase = base['cam']
    #  Disable frame readout
    cbase.RdoutCore.RdoutEn.set(0)
    #  Switch to internal triggering
    cbase.SystemRegs.TrigSrcSel.set(3)
    cbase.SystemRegs.AutoTrigEn.set(1)

def epixquad_enable(base):
    pass

def epixquad_disable(base):
    pass


# 1kfps wrappers -> reuse epixquad_* implementations
def epixquad1kfps_init(*args, **kwargs):
    return epixquad_init(*args, **kwargs)

def epixquad1kfps_init_feb(*args, **kwargs):
    return epixquad_init_feb(*args, **kwargs)

def epixquad1kfps_connectionInfo(*args, **kwargs):
    return epixquad_connectionInfo(*args, **kwargs)

def epixquad1kfps_config(*args, **kwargs):
    # keep cfgtype as passed (should be 'epixquad1kfps' from C++), your code doesn’t care
    return epixquad_config(*args, **kwargs)

def epixquad1kfps_unconfig(*args, **kwargs):
    return epixquad_unconfig(*args, **kwargs)

def epixquad1kfps_scan_keys(*args, **kwargs):
    return epixquad_scan_keys(*args, **kwargs)

def epixquad1kfps_update(*args, **kwargs):
    return epixquad_update(*args, **kwargs)

def epixquad1kfps_enable(*args, **kwargs):
    return epixquad_enable(*args, **kwargs)

def epixquad1kfps_disable(*args, **kwargs):
    return epixquad_disable(*args, **kwargs)


# EOF

import json
import logging
import os
import time
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

SHAPE = (16, 178, 192)

ENV_TEST_FILE = 'EPIXQUAD_DEBUG_TEST_FILE'
ENV_SEQUENCE_FILE = 'EPIXQUAD_DEBUG_SEQUENCE_FILE'
ENV_PATTERN_INDEX = 'EPIXQUAD_DEBUG_PATTERN_INDEX'
ENV_STEP_INDEX = 'EPIXQUAD_DEBUG_STEP_INDEX'
ENV_GROUP_INDEX = 'EPIXQUAD_DEBUG_GROUP_INDEX'
ENV_OUTDIR = 'EPIXQUAD_DEBUG_PATTERN_OUTDIR'
ENV_GROUPS = 'EPIXQUAD_DEBUG_MARKER_GROUPS'
ENV_CONTROL_FILE = 'EPIXQUAD_DEBUG_CONTROL_FILE'
DEFAULT_CONTROL_FILE = '/tmp/epixquad1kfps_pattern_control.json'


def _package_dir():
    return Path(__file__).resolve().parent


def _resolve_input_path(path_str, *, base_dir=None):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path

    candidates = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / path)
    candidates.append(Path.cwd() / path)
    candidates.append(_package_dir() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def _load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def control_file_path():
    return os.environ.get(ENV_CONTROL_FILE, DEFAULT_CONTROL_FILE)


def _load_control_request():
    """Loads optional pattern-selection state written by an external wrapper.

    This is intended for the case where the DAQ/config process is already
    running and client-side environment-variable changes will not propagate into
    that process. The wrapper updates a shared control JSON on disk before each
    Configure transition.
    """
    path = Path(control_file_path())
    if not path.exists():
        return None

    try:
        req = _load_json(path)
    except Exception as exc:
        raise ValueError(f'failed to read debug control file {path}: {exc}') from exc

    if not req.get('enabled', False):
        return None

    expires_at = req.get('expires_at')
    if expires_at is not None:
        try:
            if float(expires_at) < float(time.time()):
                logger.warning('ignoring expired debug control file %s', str(path))
                return None
        except Exception:
            pass

    req['_control_file'] = str(path.resolve())
    return req


def _coerce_pattern_index(cfg_user=None):
    env_pattern = os.environ.get(ENV_PATTERN_INDEX)
    env_step = os.environ.get(ENV_STEP_INDEX)
    cfg_pattern = None if cfg_user is None else cfg_user.get('debug_pattern_index')
    cfg_step = None if cfg_user is None else cfg_user.get('debug_test_index')

    pattern_value = cfg_pattern
    if pattern_value is None:
        pattern_value = cfg_step
    if pattern_value is None:
        pattern_value = env_pattern
    if pattern_value is None:
        pattern_value = env_step
    if pattern_value is None:
        return 0

    try:
        return int(pattern_value)
    except Exception as exc:
        raise ValueError(f'invalid debug pattern index: {pattern_value!r}') from exc


def _coerce_group_index(value=None):
    if value is None:
        value = os.environ.get(ENV_GROUP_INDEX)
    if value is None:
        return 0
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f'invalid debug group index: {value!r}') from exc


def _split_groups(group_text):
    """Normalizes marker group selection from env vars or sequence JSON.

    Accepts either a comma-separated string or a list-like object and returns
    a normalized list of non-empty group names, or None if nothing is selected.
    """
    if group_text is None:
        return None
    if isinstance(group_text, str):
        groups = [g.strip() for g in group_text.split(',') if g.strip()]
    else:
        groups = [str(g).strip() for g in group_text if str(g).strip()]
    return groups or None


def _marker_groups(test_spec):
    """Returns marker groups in first-seen order for one test JSON."""
    groups = []
    for marker in test_spec['markers']:
        group = marker.get('group')
        if group is not None and group not in groups:
            groups.append(group)
    return groups


def _select_markers(test_spec, selected_groups=None, selected_group_index=None):
    """Selects which markers from a test JSON will be materialized.

    Standalone mode:
      selected_groups comes from EPIXQUAD_DEBUG_MARKER_GROUPS.
      If it is omitted, selected_group_index chooses one marker group by
      first-seen order and defaults to group 0.

    Sequence mode:
      selected_groups comes from the sequence pattern's marker_groups field.
      If it is omitted, selected_group_index chooses one marker group by
      first-seen order and defaults to group 0.
    """
    groups_present = _marker_groups(test_spec)
    if selected_groups is None:
        if not groups_present:
            return list(test_spec['markers']), []
        group_index = _coerce_group_index(selected_group_index)
        if not (0 <= group_index < len(groups_present)):
            raise ValueError(
                f"group_index={group_index} out of range for test "
                f"{test_spec.get('test_name', 'unnamed_test')} (available groups: {groups_present})"
            )
        selected_groups = [groups_present[group_index]]
    else:
        group_index = None

    if selected_groups is None:
        return list(test_spec['markers']), groups_present

    missing = [g for g in selected_groups if g not in groups_present]
    if missing:
        raise ValueError(
            f"selected marker groups {missing} not present in test "
            f"{test_spec.get('test_name', 'unnamed_test')} (available: {groups_present})"
        )
    selected = [m for m in test_spec['markers'] if m.get('group') in selected_groups]
    return selected, selected_groups


def _materialize_test_spec(test_spec, selected_groups=None, selected_group_index=None):
    """Builds the concrete (16,178,192) pixel_map array for one selected test.

    Only markers from the selected group(s) are applied to the background-filled
    array. The returned dict contains both the materialized pixel map and the
    trbit list needed by epixquad1kfps_config.py.
    """
    shape = tuple(test_spec.get('array_shape', SHAPE))
    if shape != SHAPE:
        raise ValueError(f'expected test array shape {SHAPE}, got {shape}')

    background_value = int(test_spec['background_value'])
    default_marker_value = int(test_spec.get('default_marker_value', background_value))
    trbit_by_asic = [int(v) for v in test_spec['trbit_by_asic']]
    if len(trbit_by_asic) != SHAPE[0]:
        raise ValueError(f'expected 16 ASIC trbit entries, got {len(trbit_by_asic)}')

    arr = np.full(shape, background_value, dtype=np.uint8)
    selected_markers, selected_groups = _select_markers(
        test_spec,
        selected_groups=selected_groups,
        selected_group_index=selected_group_index,
    )
    marker_summary = []
    for marker in selected_markers:
        asic = int(marker['asic'])
        row = int(marker['row'])
        col = int(marker['col'])
        value = int(marker.get('value', default_marker_value))

        if not (0 <= asic < SHAPE[0]):
            raise ValueError(f'invalid marker ASIC index {asic}')
        if not (0 <= row < SHAPE[1]):
            raise ValueError(f'invalid marker row {row}')
        if not (0 <= col < SHAPE[2]):
            raise ValueError(f'invalid marker col {col}')

        arr[asic, row, col] = value
        marker_summary.append({
            'label': marker.get('label', ''),
            'group': marker.get('group'),
            'asic': asic,
            'row': row,
            'col': col,
            'value': value,
        })

    return {
        'pixel_map': arr,
        'trbit_by_asic': trbit_by_asic,
        'test_name': test_spec.get('test_name', 'unnamed_test'),
        'description': test_spec.get('description', ''),
        'selected_groups': selected_groups or [],
        'selected_markers': marker_summary,
        'active_pixel_count': len(marker_summary),
        'background_value': background_value,
    }


def _load_test_file(test_file, selected_groups=None, selected_group_index=None):
    """Loads one standalone test JSON and materializes the selected group(s)."""
    test_path = _resolve_input_path(test_file)
    test_spec = _load_json(test_path)
    materialized = _materialize_test_spec(
        test_spec,
        selected_groups=selected_groups,
        selected_group_index=selected_group_index,
    )
    materialized['source_file'] = str(test_path)
    materialized['source_kind'] = 'test'
    return materialized


def _sequence_patterns(sequence_spec):
    return sequence_spec.get('patterns', sequence_spec.get('steps', []))


def _pattern_index(entry, default_index):
    if 'pattern_index' in entry:
        return int(entry['pattern_index'])
    if 'step_index' in entry:
        return int(entry['step_index'])
    return int(default_index)


def _load_sequence_pattern(sequence_file, pattern_index):
    """Loads one sequence pattern and materializes its selected marker group(s)."""
    seq_path = _resolve_input_path(sequence_file)
    sequence_spec = _load_json(seq_path)
    patterns = _sequence_patterns(sequence_spec)
    matches = [
        pattern for i, pattern in enumerate(patterns)
        if _pattern_index(pattern, i) == int(pattern_index)
    ]
    if not matches:
        raise ValueError(f'pattern_index={pattern_index} not found in sequence {seq_path}')
    if len(matches) > 1:
        raise ValueError(f'duplicate pattern_index={pattern_index} in sequence {seq_path}')

    pattern = matches[0]
    test_path = _resolve_input_path(pattern['test_file'], base_dir=seq_path.parent)
    test_spec = _load_json(test_path)
    selected_groups = _split_groups(pattern.get('marker_groups'))
    selected_group_index = pattern.get('group_index', 0)
    materialized = _materialize_test_spec(
        test_spec,
        selected_groups=selected_groups,
        selected_group_index=selected_group_index,
    )
    materialized.update({
        'pattern_index': int(pattern_index),
        'pattern_label': pattern.get('label', f'pattern-{pattern_index}'),
        'sequence_name': sequence_spec.get('sequence_name', 'unnamed_sequence'),
        'source_file': str(seq_path),
        'source_kind': 'sequence',
        'test_file': str(test_path),
    })
    return materialized


def load_debug_pattern(cfg_user=None):
    """Entry point used by epixquad1kfps_config.py.

    Selection logic:
      1. If EPIXQUAD_DEBUG_TEST_FILE is set, load that test file directly.
         In this mode marker groups come from EPIXQUAD_DEBUG_MARKER_GROUPS, or
         from EPIXQUAD_DEBUG_GROUP_INDEX if group names are not provided. The
         default is group 0 in first-seen order.
      2. Else if EPIXQUAD_DEBUG_SEQUENCE_FILE is set, use
         EPIXQUAD_DEBUG_PATTERN_INDEX (or the legacy EPIXQUAD_DEBUG_STEP_INDEX)
         to select one sequence pattern. In this mode marker groups come from
         that pattern's marker_groups field, or fall back to group_index=0 if
         marker_groups is omitted.
      3. Else return None and do nothing.
    """
    test_file = os.environ.get(ENV_TEST_FILE)
    sequence_file = os.environ.get(ENV_SEQUENCE_FILE)
    control_req = None if (test_file or sequence_file) else _load_control_request()

    if test_file and sequence_file:
        raise ValueError(f'use only one of {ENV_TEST_FILE} or {ENV_SEQUENCE_FILE}')

    if test_file or sequence_file:
        pattern_index = _coerce_pattern_index(cfg_user)
        if test_file:
            selected_groups = _split_groups(os.environ.get(ENV_GROUPS))
            selected_group_index = None if selected_groups is not None else _coerce_group_index()
            materialized = _load_test_file(
                test_file,
                selected_groups=selected_groups,
                selected_group_index=selected_group_index,
            )
            materialized['pattern_index'] = pattern_index
        else:
            materialized = _load_sequence_pattern(sequence_file, pattern_index)
        selection_source = 'env'
        outdir = os.environ.get(ENV_OUTDIR)
    elif control_req is not None:
        if control_req.get('test_file') and control_req.get('sequence_file'):
            raise ValueError('control file must specify only one of test_file or sequence_file')
        if control_req.get('test_file'):
            selected_groups = _split_groups(control_req.get('marker_groups'))
            selected_group_index = None if selected_groups is not None else _coerce_group_index(control_req.get('group_index', 0))
            materialized = _load_test_file(
                control_req['test_file'],
                selected_groups=selected_groups,
                selected_group_index=selected_group_index,
            )
            materialized['pattern_index'] = int(control_req.get('pattern_index', control_req.get('step_index', 0)))
        elif control_req.get('sequence_file'):
            materialized = _load_sequence_pattern(
                control_req['sequence_file'],
                int(control_req.get('pattern_index', control_req.get('step_index', 0))),
            )
        else:
            raise ValueError('control file must specify test_file or sequence_file')
        materialized['control_file'] = control_req['_control_file']
        selection_source = 'control_file'
        outdir = control_req.get('pattern_outdir')
    else:
        return None

    materialized['selection_source'] = selection_source

    if outdir:
        outpath = _resolve_input_path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        stem = materialized['test_name']
        if materialized['source_kind'] == 'sequence':
            stem = f"{materialized['sequence_name']}_pattern{materialized['pattern_index']:02d}_{stem}"
        np.save(outpath / f'{stem}.npy', materialized['pixel_map'])
        meta = {
            'source_kind': materialized['source_kind'],
            'source_file': materialized['source_file'],
            'test_name': materialized['test_name'],
            'description': materialized['description'],
            'pattern_index': materialized.get('pattern_index', 0),
            'selected_groups': materialized.get('selected_groups', []),
            'active_pixel_count': materialized.get('active_pixel_count', 0),
            'background_value': materialized.get('background_value'),
            'selected_markers': materialized.get('selected_markers', []),
            'trbit_by_asic': materialized.get('trbit_by_asic', []),
            'selection_source': materialized.get('selection_source', 'unknown'),
        }
        if 'sequence_name' in materialized:
            meta['sequence_name'] = materialized['sequence_name']
            meta['pattern_label'] = materialized.get('pattern_label', '')
        if 'test_file' in materialized:
            meta['test_file'] = materialized['test_file']
        if 'control_file' in materialized:
            meta['control_file'] = materialized['control_file']
        with (outpath / f'{stem}.json').open('w') as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    return materialized

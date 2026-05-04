#!/usr/bin/env python3

"""Validate ePix quad debug-pattern runs against raw data.

This implements the first two phases of the detector-day validation workflow.

Pass 0:
  Build the expected run-to-pattern table from either:
    - one pattern-sequence JSON in the original marker/test-file schema, or
    - one direct-write JSON in the newer bank-probe schema,
  plus a starting run number.

Pass 2:
  For each selected run, scan ``det.raw.raw(evt)`` and compute stable raw-word
  summaries that can later be used to infer the empirical mapping from logical
  ``(asic, row, col)`` markers to observed raw coordinates.

Primary outputs:
  - ``run_table.json`` / ``run_table.csv``
  - one subdirectory per run with:
      ``bit14_occupancy.npy``
      ``background_deviation.npy``
      ``dominant_code.npy``
      ``dominant_confidence.npy``
      ``summary.json``

The current detector-day patterns use sparse fixed-mode markers, so this script
does not try to fully infer geometry yet. Instead, it records:
  - which raw top-bit code dominates each pixel,
  - how strongly that pixel deviates from the run's background code,
  - top candidate raw coordinates ranked by that deviation.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np


B14 = 0x4000
RAW_SHAPE = (4, 352, 384)


logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Build run tables and raw-bit summaries for epixquad1kfps debug-pattern runs'
    )
    parser.add_argument('--sequence', required=True,
                        help='Pattern-sequence JSON file or direct-write JSON file')
    parser.add_argument('--exp', required=True,
                        help='Experiment name, for example ued1016014')
    parser.add_argument('--run-start', required=True, type=int,
                        help='First DAQ run number corresponding to pattern_index 0')
    parser.add_argument('--outdir', required=True,
                        help='Output directory for run table and per-run summaries')
    parser.add_argument('--detector', default='epixquad1kfps',
                        help='Detector name for psana (default: epixquad1kfps)')
    parser.add_argument('--xtc-dir', default=None,
                        help='Optional XTC directory passed to psana DataSource')
    parser.add_argument('--events', type=int, default=None,
                        help='Maximum number of events to analyze per run. Defaults to sequence defaults.readout_count or 50.')
    parser.add_argument('--patterns', default=None,
                        help='Comma-separated subset of pattern_index values to analyze')
    parser.add_argument('--candidate-threshold', type=float, default=0.50,
                        help='Minimum background-deviation score for candidate pixels (default 0.50)')
    parser.add_argument('--confidence-threshold', type=float, default=0.80,
                        help='Minimum dominant-code confidence for candidate pixels (default 0.80)')
    parser.add_argument('--topk', type=int, default=16,
                        help='Number of top candidate raw pixels to record per run (default 16)')
    parser.add_argument('--pass0-only', action='store_true',
                        help='Only write the run table; do not read data')
    parser.add_argument('--log-level', default='INFO',
                        help='Python logging level (default INFO)')
    return parser.parse_args()


def _resolve_input_path(path_str, *, base_dir=None):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        candidate = (Path(base_dir) / path).resolve()
        if candidate.exists():
            return candidate
    return path.resolve()


def _load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def _sequence_patterns(sequence_spec):
    return sequence_spec.get('patterns', sequence_spec.get('steps', []))


def _pattern_index(pattern, default_index):
    if 'pattern_index' in pattern:
        return int(pattern['pattern_index'])
    if 'step_index' in pattern:
        return int(pattern['step_index'])
    return int(default_index)


def _selected_patterns(sequence_spec, patterns_arg):
    patterns = sorted(
        _sequence_patterns(sequence_spec),
        key=lambda entry: _pattern_index(entry, 0),
    )
    if not patterns_arg:
        return patterns
    selected = {int(v.strip()) for v in patterns_arg.split(',') if v.strip()}
    return [
        pattern for i, pattern in enumerate(patterns)
        if _pattern_index(pattern, i) in selected
    ]


def _split_groups(group_text):
    if group_text is None:
        return None
    if isinstance(group_text, str):
        groups = [g.strip() for g in group_text.split(',') if g.strip()]
    else:
        groups = [str(g).strip() for g in group_text if str(g).strip()]
    return groups or None


def _marker_groups(test_spec):
    groups = []
    for marker in test_spec['markers']:
        group = marker.get('group')
        if group is not None and group not in groups:
            groups.append(group)
    return groups


def _selected_markers(test_spec, pattern):
    selected_groups = _split_groups(pattern.get('marker_groups'))
    groups_present = _marker_groups(test_spec)
    if selected_groups is None:
        group_index = int(pattern.get('group_index', 0))
        if not groups_present:
            return list(test_spec['markers']), []
        if not (0 <= group_index < len(groups_present)):
            raise ValueError(
                f'group_index={group_index} out of range for test {test_spec.get("test_name", "unnamed_test")} '
                f'(available groups: {groups_present})'
            )
        selected_groups = [groups_present[group_index]]

    selected = [marker for marker in test_spec['markers'] if marker.get('group') in selected_groups]
    return selected, selected_groups


def _build_run_table(sequence_path, sequence_spec, run_start, patterns):
    sequence_name = sequence_spec.get(
        'sequence_name',
        sequence_spec.get('direct_name', 'unnamed_sequence'),
    )
    rows = []
    for i, pattern in enumerate(patterns):
        pattern_index = _pattern_index(pattern, i)
        run_number = run_start + pattern_index
        if 'test_file' in pattern:
            test_path = _resolve_input_path(pattern['test_file'], base_dir=sequence_path.parent)
            test_spec = _load_json(test_path)
            selected_markers, selected_groups = _selected_markers(test_spec, pattern)
            rows.append({
                'sequence_name': sequence_name,
                'pattern_index': pattern_index,
                'pattern_label': pattern.get('label', f'pattern-{pattern_index}'),
                'run': run_number,
                'test_name': test_spec.get('test_name', 'unnamed_test'),
                'test_file': str(test_path),
                'marker_groups': selected_groups,
                'markers': [
                    {
                        'label': marker.get('label', ''),
                        'group': marker.get('group'),
                        'asic': int(marker['asic']),
                        'row': int(marker['row']),
                        'col': int(marker['col']),
                        'value': int(marker.get('value', test_spec.get('default_marker_value', 0))),
                        'tags': marker.get('tags', []),
                    }
                    for marker in selected_markers
                ],
                'purpose': pattern.get('purpose', ''),
                'priority': pattern.get('priority', ''),
                'notes': pattern.get('notes', []),
            })
            continue

        if 'ops' in pattern:
            coordinate_mode = pattern.get(
                'coordinate_mode',
                sequence_spec.get('coordinate_mode', 'bank_rc_178x48'),
            )
            ops = pattern.get('ops', [])
            pixel_ops = [op for op in ops if op.get('kind', 'pixel') == 'pixel']
            markers = [
                {
                    'label': f"op{idx}",
                    'group': f"bank{int(op['bank'])}",
                    'asic': int(op['asic']),
                    'row': int(op['row']),
                    'col': int(op['col']),
                    'value': int(op.get('value', sequence_spec.get('default_selected_value', 0))),
                    'tags': ['direct-write', coordinate_mode],
                }
                for idx, op in enumerate(pixel_ops)
            ]
            bank_ids = sorted({int(op['bank']) for op in ops if 'bank' in op})
            marker_groups = [f'bank{bank}' for bank in bank_ids]
            rows.append({
                'sequence_name': sequence_name,
                'pattern_index': pattern_index,
                'pattern_label': pattern.get('label', f'pattern-{pattern_index}'),
                'run': run_number,
                'test_name': sequence_spec.get('direct_name', 'unnamed_direct'),
                'test_file': str(sequence_path),
                'marker_groups': marker_groups,
                'markers': markers,
                'purpose': pattern.get('purpose', sequence_spec.get('description', '')),
                'priority': pattern.get('priority', ''),
                'notes': pattern.get('notes', []),
                'coordinate_mode': coordinate_mode,
                'bank_index': bank_ids[0] if len(bank_ids) == 1 else None,
                'direct_ops': ops,
            })
            continue

        raise KeyError(
            f'pattern_index={pattern_index} in {sequence_path} has neither test_file nor ops'
        )
    return rows


def _write_run_table(outdir, run_table):
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / 'run_table.json').open('w') as f:
        json.dump(run_table, f, indent=2, sort_keys=True)

    with (outdir / 'run_table.csv').open('w', newline='') as f:
        fieldnames = [
            'sequence_name',
            'pattern_index',
            'pattern_label',
            'run',
            'test_name',
            'test_file',
            'marker_groups',
            'purpose',
            'priority',
            'marker_count',
            'marker_labels',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_table:
            writer.writerow({
                'sequence_name': row['sequence_name'],
                'pattern_index': row['pattern_index'],
                'pattern_label': row['pattern_label'],
                'run': row['run'],
                'test_name': row['test_name'],
                'test_file': row['test_file'],
                'marker_groups': ','.join(row['marker_groups']),
                'purpose': row['purpose'],
                'priority': row['priority'],
                'marker_count': len(row['markers']),
                'marker_labels': ','.join(m['label'] for m in row['markers']),
            })


def _resolve_detector_name(run, detector_name):
    if detector_name:
        return detector_name
    candidates = sorted(name for name in run.detnames if 'epixquad' in name.lower())
    if not candidates:
        raise RuntimeError(f'no detector name containing "epixquad" found in run.detnames={sorted(run.detnames)}')
    return candidates[0]


def _candidate_table(background_deviation, dominant_code, dominant_confidence, bit14_occupancy,
                     candidate_threshold, confidence_threshold, topk):
    mask = np.logical_and(
        background_deviation >= candidate_threshold,
        dominant_confidence >= confidence_threshold,
    )
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []

    scored = []
    for module, row, col in coords:
        scored.append({
            'module': int(module),
            'row': int(row),
            'col': int(col),
            'background_deviation': float(background_deviation[module, row, col]),
            'dominant_code': int(dominant_code[module, row, col]),
            'dominant_confidence': float(dominant_confidence[module, row, col]),
            'bit14_occupancy': float(bit14_occupancy[module, row, col]),
        })

    scored.sort(
        key=lambda entry: (
            -entry['background_deviation'],
            -entry['dominant_confidence'],
            entry['module'],
            entry['row'],
            entry['col'],
        )
    )
    return scored[:topk]


def _coarse_scores(background_deviation):
    module_scores = np.sum(background_deviation, axis=(1, 2), dtype=np.float64)
    asic_scores = np.zeros((background_deviation.shape[0], 2, 2), dtype=np.float64)
    for module in range(background_deviation.shape[0]):
        for rblk in range(2):
            for cblk in range(2):
                y0 = rblk * 176
                y1 = y0 + 176
                x0 = cblk * 192
                x1 = x0 + 192
                asic_scores[module, rblk, cblk] = np.sum(
                    background_deviation[module, y0:y1, x0:x1],
                    dtype=np.float64,
                )
    return module_scores, asic_scores


def _save_run_summary(run_dir, summary, arrays):
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / 'bit14_occupancy.npy', arrays['bit14_occupancy'])
    np.save(run_dir / 'background_deviation.npy', arrays['background_deviation'])
    np.save(run_dir / 'dominant_code.npy', arrays['dominant_code'])
    np.save(run_dir / 'dominant_confidence.npy', arrays['dominant_confidence'])

    with (run_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _analyze_run(entry, args, default_events):
    from psana import DataSource

    ds_kwargs = {
        'exp': args.exp,
        'run': entry['run'],
    }
    if args.xtc_dir is not None:
        ds_kwargs['dir'] = args.xtc_dir
    if default_events is not None:
        ds_kwargs['max_events'] = default_events

    logger.info('analyzing run=%d pattern=%02d label=%s',
                entry['run'], entry['pattern_index'], entry['pattern_label'])
    ds = DataSource(**ds_kwargs)
    run = next(ds.runs())
    detector_name = _resolve_detector_name(run, args.detector)
    det = run.Detector(detector_name)

    n_events = 0
    n_valid = 0
    code_counts = None
    bit14_sum = None
    raw_shape = None

    for evt in run.events():
        n_events += 1
        raw = det.raw.raw(evt)
        if raw is None:
            continue
        raw = np.asarray(raw)
        if raw_shape is None:
            raw_shape = tuple(raw.shape)
            if raw_shape != RAW_SHAPE:
                raise RuntimeError(f'expected raw shape {RAW_SHAPE}, got {raw_shape} for run {entry["run"]}')
            code_counts = np.zeros((4,) + raw_shape, dtype=np.uint32)
            bit14_sum = np.zeros(raw_shape, dtype=np.uint32)

        codes = np.right_shift(raw, 14) & 0x3
        for code in range(4):
            code_counts[code] += (codes == code)
        bit14_sum += ((raw & B14) != 0)
        n_valid += 1

    if n_valid == 0 or code_counts is None or bit14_sum is None:
        raise RuntimeError(f'no valid raw frames found for run {entry["run"]}')

    bit14_occupancy = np.asarray(bit14_sum, dtype=np.float32) / float(n_valid)
    dominant_code = np.argmax(code_counts, axis=0).astype(np.uint8)
    dominant_counts = np.max(code_counts, axis=0)
    dominant_confidence = np.asarray(dominant_counts, dtype=np.float32) / float(n_valid)
    global_code_counts = np.sum(code_counts, axis=(1, 2, 3), dtype=np.uint64)
    background_code = int(np.argmax(global_code_counts))
    background_fraction = float(global_code_counts[background_code]) / float(n_valid * np.prod(raw_shape))
    background_deviation = 1.0 - (
        np.asarray(code_counts[background_code], dtype=np.float32) / float(n_valid)
    )

    candidate_pixels = _candidate_table(
        background_deviation,
        dominant_code,
        dominant_confidence,
        bit14_occupancy,
        args.candidate_threshold,
        args.confidence_threshold,
        args.topk,
    )
    module_scores, asic_scores = _coarse_scores(background_deviation)

    summary = {
        'exp': args.exp,
        'run': entry['run'],
        'detector': detector_name,
        'pattern_index': entry['pattern_index'],
        'pattern_label': entry['pattern_label'],
        'test_name': entry['test_name'],
        'marker_groups': entry['marker_groups'],
        'markers': entry['markers'],
        'events_seen': n_events,
        'events_used': n_valid,
        'background_code': background_code,
        'background_fraction': background_fraction,
        'global_code_counts': [int(v) for v in global_code_counts.tolist()],
        'module_scores': [float(v) for v in module_scores.tolist()],
        'asic_scores': np.asarray(asic_scores, dtype=np.float64).tolist(),
        'candidate_threshold': float(args.candidate_threshold),
        'confidence_threshold': float(args.confidence_threshold),
        'top_candidates': candidate_pixels,
    }
    if 'coordinate_mode' in entry:
        summary['coordinate_mode'] = entry['coordinate_mode']
    if 'bank_index' in entry:
        summary['bank_index'] = entry['bank_index']
    if 'direct_ops' in entry:
        summary['direct_ops'] = entry['direct_ops']
    arrays = {
        'bit14_occupancy': bit14_occupancy,
        'background_deviation': background_deviation.astype(np.float32),
        'dominant_code': dominant_code,
        'dominant_confidence': dominant_confidence.astype(np.float32),
    }
    return summary, arrays


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format='[%(levelname).1s] %(message)s',
    )

    sequence_path = _resolve_input_path(args.sequence)
    sequence_spec = _load_json(sequence_path)
    outdir = Path(args.outdir).expanduser().resolve()
    patterns = _selected_patterns(sequence_spec, args.patterns)
    if not patterns:
        raise RuntimeError('no patterns selected')

    run_table = _build_run_table(sequence_path, sequence_spec, args.run_start, patterns)
    _write_run_table(outdir, run_table)
    logger.info('wrote run table with %d pattern rows to %s', len(run_table), str(outdir))

    if args.pass0_only:
        return

    default_events = args.events
    if default_events is None:
        default_events = int(sequence_spec.get('defaults', {}).get('readout_count', 50))

    run_summaries = []
    for entry in run_table:
        summary, arrays = _analyze_run(entry, args, default_events)
        run_dir = outdir / ('run%04d_pattern%02d' % (entry['run'], entry['pattern_index']))
        _save_run_summary(run_dir, summary, arrays)
        run_summaries.append(summary)

    with (outdir / 'run_summaries.json').open('w') as f:
        json.dump(run_summaries, f, indent=2, sort_keys=True)
    logger.info('wrote %d run summaries to %s', len(run_summaries), str(outdir / 'run_summaries.json'))


if __name__ == '__main__':
    main()

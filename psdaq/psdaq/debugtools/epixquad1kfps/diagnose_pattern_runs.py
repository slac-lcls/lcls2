#!/usr/bin/env python3

"""Diagnose extracted ePix quad debug-pattern runs.

This is the first diagnosis layer on top of ``validate_pattern_runs.py``.
Currently implemented diagnoses:

  - ``module_location``:
      rank raw modules and coarse ASIC-sized blocks (176x192) within them
  - ``asic_orientation``:
      reuse the coarse module/block result, then compare the observed 2-point
      candidate vector against the intended logical marker vector and classify
      the best transform as ``identity``, ``flipud``, ``fliplr``, ``rot180``,
      or ``unresolved``
  - ``quadrants``:
      reuse the coarse module/block result, then rank the four quadrant-sized
      regions (88x96) inside the winning coarse block
  - ``column_regions``:
      reuse the coarse module/block result, then rank the four 48-column bands
      inside the winning coarse block
  - ``row_regions``:
      reuse the coarse module/block result, then rank the four row bands
      inside the winning coarse block
  - ``bank_marker_orientation``:
      reuse the coarse module/block result, then inspect one expected bank
      region for the two strongest sparse-marker peaks and classify the
      in-bank transform as ``identity``, ``flipud``, ``fliplr``, ``rot180``,
      or ``unresolved``

The diagnosis operates on previously extracted products and does not reread raw
data. This keeps diagnosis cheap and easy to iterate on.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

BLOCK_LABELS = {
    (0, 0): 'TL',
    (0, 1): 'TR',
    (1, 0): 'BL',
    (1, 1): 'BR',
}

ORIENTATION_TRANSFORMS = {
    'identity': lambda dr, dc: (dr, dc),
    'flipud': lambda dr, dc: (-dr, dc),
    'fliplr': lambda dr, dc: (dr, -dc),
    'rot180': lambda dr, dc: (-dr, -dc),
}

BANK_COORD_MODES = ('bank_rc_178x48', 'bank_rc_44x192')
BANK_MARKER_TEMPLATES = {
    'bank_rc_178x48': ((12, 7), (145, 35)),
    'bank_rc_44x192': ((6, 20), (31, 150)),
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose extracted epixquad1kfps debug-pattern runs'
    )
    parser.add_argument('--input-dir', required=True,
                        help='Directory created by validate_pattern_runs.py')
    parser.add_argument('--diag', default='module_location',
                        help='Diagnosis mode to run (module_location, asic_orientation, quadrants, column_regions, row_regions, or bank_marker_orientation)')
    parser.add_argument('--run', default=None,
                        help='Comma-separated subset of run numbers to diagnose')
    parser.add_argument('--runs', default=None,
                        help='Deprecated alias for --run')
    parser.add_argument('--patterns', default=None,
                        help='Comma-separated subset of pattern_index values to diagnose')
    parser.add_argument('--min-total-score', type=float, default=0.0,
                        help='Warn if the total coarse deviation score is at or below this value')
    parser.add_argument('--log-level', default='INFO',
                        help='Python logging level (default INFO)')
    parser.add_argument('--coordinate-mode', default='bank_rc_178x48',
                        help='Bank coordinate convention for bank_marker_orientation '
                             '(bank_rc_178x48 or bank_rc_44x192)')
    parser.add_argument('--bank-index', type=int, default=None,
                        help='Logical bank index 0..3 for bank_marker_orientation')
    parser.add_argument('--peak-exclusion-radius', type=int, default=6,
                        help='Suppression radius in pixels when searching for two sparse-marker peaks')
    return parser.parse_args()


def _load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def _parse_int_set(text):
    if text is None:
        return None
    vals = {int(v.strip()) for v in text.split(',') if v.strip()}
    return vals or None


def _selected_rows(run_summaries, runs_arg, patterns_arg):
    run_filter = _parse_int_set(runs_arg)
    pattern_filter = _parse_int_set(patterns_arg)
    rows = []
    for row in run_summaries:
        if run_filter is not None and int(row['run']) not in run_filter:
            continue
        if pattern_filter is not None and int(row['pattern_index']) not in pattern_filter:
            continue
        rows.append(row)
    return rows


def _safe_ratio(num, den):
    if den <= 0:
        return None
    return float(num) / float(den)


def _ranked_modules(module_scores):
    indexed = [
        {'module': int(module), 'score': float(score)}
        for module, score in enumerate(module_scores)
    ]
    indexed.sort(key=lambda item: (-item['score'], item['module']))
    return indexed


def _ranked_blocks(asic_scores_for_module):
    ranked = []
    for rblk in range(asic_scores_for_module.shape[0]):
        for cblk in range(asic_scores_for_module.shape[1]):
            ranked.append({
                'block_row': int(rblk),
                'block_col': int(cblk),
                'block_label': BLOCK_LABELS[(rblk, cblk)],
                'score': float(asic_scores_for_module[rblk, cblk]),
            })
    ranked.sort(key=lambda item: (-item['score'], item['block_row'], item['block_col']))
    return ranked


def _coarse_bbox(module, block_row, block_col):
    y0 = block_row * 176
    y1 = y0 + 176
    x0 = block_col * 192
    x1 = x0 + 192
    return {
        'module': int(module),
        'row0': int(y0),
        'row1': int(y1),
        'col0': int(x0),
        'col1': int(x1),
    }


def _load_run_array(input_dir, row, name):
    run_dir = input_dir / ('run%04d_pattern%02d' % (int(row['run']), int(row['pattern_index'])))
    path = run_dir / name
    if not path.exists():
        raise RuntimeError(f'missing extracted array {path}')
    return np.load(path)


def _inside_bbox(candidate, bbox):
    return (
        int(candidate['module']) == int(bbox['module']) and
        int(bbox['row0']) <= int(candidate['row']) < int(bbox['row1']) and
        int(bbox['col0']) <= int(candidate['col']) < int(bbox['col1'])
    )


def _pair_candidates(module_diag):
    top_candidates = module_diag.get('top_candidates', [])
    bbox = module_diag['best_block']['bbox_raw']
    in_block = [cand for cand in top_candidates if _inside_bbox(cand, bbox)]
    if len(in_block) >= 2:
        return in_block[:2], 'best_block'

    best_module = int(module_diag['best_module']['module'])
    in_module = [cand for cand in top_candidates if int(cand['module']) == best_module]
    if len(in_module) >= 2:
        return in_module[:2], 'best_module'

    if len(top_candidates) >= 2:
        return top_candidates[:2], 'global'

    return [], 'insufficient_candidates'


def _logical_marker_pair(markers):
    if len(markers) < 2:
        return None
    marker_a = markers[0]
    marker_b = markers[1]
    return {
        'marker_a': marker_a,
        'marker_b': marker_b,
        'vector_row': int(marker_b['row']) - int(marker_a['row']),
        'vector_col': int(marker_b['col']) - int(marker_a['col']),
    }


def _orientation_match(expected_vec, observed_vec):
    dr, dc = int(expected_vec[0]), int(expected_vec[1])
    orow, ocol = int(observed_vec[0]), int(observed_vec[1])
    ranked = []
    for name, transform in ORIENTATION_TRANSFORMS.items():
        trow, tcol = transform(dr, dc)
        err_l1 = abs(orow - trow) + abs(ocol - tcol)
        exact = (orow == trow and ocol == tcol)
        ranked.append({
            'operator': name,
            'expected_vector': [int(trow), int(tcol)],
            'observed_vector': [int(orow), int(ocol)],
            'error_l1': int(err_l1),
            'exact': bool(exact),
        })
    ranked.sort(key=lambda item: (item['error_l1'], item['operator']))
    return ranked


def _bank_bbox_in_block(bbox, coordinate_mode, bank_index):
    module = int(bbox['module'])
    if coordinate_mode == 'bank_rc_178x48':
        x0 = int(bbox['col0']) + int(bank_index) * 48
        x1 = x0 + 48
        return {
            'module': module,
            'row0': int(bbox['row0']),
            'row1': int(bbox['row1']),
            'col0': int(x0),
            'col1': int(x1),
        }
    if coordinate_mode == 'bank_rc_44x192':
        y0 = int(bbox['row0']) + int(bank_index) * 44
        y1 = y0 + 44
        return {
            'module': module,
            'row0': int(y0),
            'row1': int(y1),
            'col0': int(bbox['col0']),
            'col1': int(bbox['col1']),
        }
    raise ValueError(f'unsupported coordinate_mode {coordinate_mode!r}')


def _extract_two_peaks(background_deviation, bbox, exclusion_radius):
    module = int(bbox['module'])
    arr = np.asarray(
        background_deviation[module, bbox['row0']:bbox['row1'], bbox['col0']:bbox['col1']],
        dtype=np.float64,
    )
    work = np.array(arr, copy=True)
    peaks = []
    for _ in range(2):
        flat_idx = int(np.argmax(work))
        score = float(work.flat[flat_idx])
        if score <= 0:
            break
        local_row, local_col = np.unravel_index(flat_idx, work.shape)
        peaks.append({
            'module': module,
            'row': int(bbox['row0'] + local_row),
            'col': int(bbox['col0'] + local_col),
            'local_row': int(local_row),
            'local_col': int(local_col),
            'score': score,
        })
        y0 = max(0, local_row - exclusion_radius)
        y1 = min(work.shape[0], local_row + exclusion_radius + 1)
        x0 = max(0, local_col - exclusion_radius)
        x1 = min(work.shape[1], local_col + exclusion_radius + 1)
        work[y0:y1, x0:x1] = 0.0
    return peaks


def _diagnose_bank_marker_orientation(row, input_dir, min_total_score, coordinate_mode, bank_index, exclusion_radius):
    if coordinate_mode not in BANK_COORD_MODES:
        raise ValueError(
            f'unsupported coordinate_mode {coordinate_mode!r}; expected one of {BANK_COORD_MODES}'
        )
    if bank_index is None or not (0 <= int(bank_index) < 4):
        raise ValueError('bank_marker_orientation requires --bank-index in the range 0..3')

    module_diag = _diagnose_module_location(row, min_total_score)
    background_deviation = _load_run_array(input_dir, row, 'background_deviation.npy')
    coarse_bbox = module_diag['best_block']['bbox_raw']
    bank_bbox = _bank_bbox_in_block(coarse_bbox, coordinate_mode, int(bank_index))
    peaks = _extract_two_peaks(background_deviation, bank_bbox, exclusion_radius)
    expected_a, expected_b = BANK_MARKER_TEMPLATES[coordinate_mode]
    expected_vec = (
        int(expected_b[0]) - int(expected_a[0]),
        int(expected_b[1]) - int(expected_a[1]),
    )

    warnings = list(module_diag.get('warnings', []))
    if len(peaks) < 2:
        warnings.append('insufficient_bank_marker_peaks')

    best_match = None
    alternative_ordering = None
    if len(peaks) >= 2:
        p0, p1 = peaks[0], peaks[1]
        observed_forward = (
            int(p1['local_row']) - int(p0['local_row']),
            int(p1['local_col']) - int(p0['local_col']),
        )
        observed_reverse = (
            int(p0['local_row']) - int(p1['local_row']),
            int(p0['local_col']) - int(p1['local_col']),
        )
        forward_ranked = _orientation_match(expected_vec, observed_forward)
        reverse_ranked = _orientation_match(expected_vec, observed_reverse)
        best_forward = forward_ranked[0]
        best_reverse = reverse_ranked[0]
        if best_forward['error_l1'] <= best_reverse['error_l1']:
            best_match = {
                'operator': best_forward['operator'],
                'error_l1': best_forward['error_l1'],
                'exact': best_forward['exact'],
                'observed_order': ['peak0', 'peak1'],
                'expected_points': [list(expected_a), list(expected_b)],
                'observed_vector': best_forward['observed_vector'],
                'expected_vector_for_operator': best_forward['expected_vector'],
                'peak_points': [p0, p1],
                'all_operator_matches': forward_ranked,
            }
            alternative_ordering = {
                'observed_order': ['peak1', 'peak0'],
                'best_operator': best_reverse['operator'],
                'error_l1': best_reverse['error_l1'],
                'all_operator_matches': reverse_ranked,
            }
        else:
            best_match = {
                'operator': best_reverse['operator'],
                'error_l1': best_reverse['error_l1'],
                'exact': best_reverse['exact'],
                'observed_order': ['peak1', 'peak0'],
                'expected_points': [list(expected_a), list(expected_b)],
                'observed_vector': best_reverse['observed_vector'],
                'expected_vector_for_operator': best_reverse['expected_vector'],
                'peak_points': [p1, p0],
                'all_operator_matches': reverse_ranked,
            }
            alternative_ordering = {
                'observed_order': ['peak0', 'peak1'],
                'best_operator': best_forward['operator'],
                'error_l1': best_forward['error_l1'],
                'all_operator_matches': forward_ranked,
            }
        if best_match['error_l1'] != 0:
            warnings.append('bank_marker_nonexact_match')

    diagnosis = dict(module_diag)
    diagnosis.update({
        'diagnosis_type': 'bank_marker_orientation',
        'coordinate_mode': coordinate_mode,
        'bank_index': int(bank_index),
        'bank_bbox_raw': bank_bbox,
        'expected_marker_points': [list(expected_a), list(expected_b)],
        'peak_exclusion_radius': int(exclusion_radius),
        'bank_marker_orientation': best_match if best_match is not None else {
            'operator': 'unresolved',
            'error_l1': None,
            'exact': False,
            'observed_order': [],
            'expected_points': [list(expected_a), list(expected_b)],
            'observed_vector': None,
            'expected_vector_for_operator': None,
            'peak_points': peaks,
            'all_operator_matches': [],
        },
        'alternative_ordering': alternative_ordering,
    })
    diagnosis['warnings'] = warnings
    return diagnosis


def _diagnose_asic_orientation(row, min_total_score):
    module_diag = _diagnose_module_location(row, min_total_score)
    logical_pair = _logical_marker_pair(module_diag.get('markers', []))
    candidates, candidate_scope = _pair_candidates(module_diag)

    warnings = list(module_diag.get('warnings', []))
    if logical_pair is None:
        warnings.append('insufficient_logical_markers')
    if len(candidates) < 2:
        warnings.append('insufficient_observed_candidates')

    best_match = None
    alternative_ordering = None
    if logical_pair is not None and len(candidates) >= 2:
        c0, c1 = candidates[0], candidates[1]
        observed_forward = (
            int(c1['row']) - int(c0['row']),
            int(c1['col']) - int(c0['col']),
        )
        observed_reverse = (
            int(c0['row']) - int(c1['row']),
            int(c0['col']) - int(c1['col']),
        )
        forward_ranked = _orientation_match(
            (logical_pair['vector_row'], logical_pair['vector_col']),
            observed_forward,
        )
        reverse_ranked = _orientation_match(
            (logical_pair['vector_row'], logical_pair['vector_col']),
            observed_reverse,
        )
        best_forward = forward_ranked[0]
        best_reverse = reverse_ranked[0]
        if (best_forward['error_l1'], candidates[0]['module'], candidates[0]['row'], candidates[0]['col']) <= \
           (best_reverse['error_l1'], candidates[1]['module'], candidates[1]['row'], candidates[1]['col']):
            best_match = {
                'operator': best_forward['operator'],
                'error_l1': best_forward['error_l1'],
                'exact': best_forward['exact'],
                'observed_order': ['candidate0', 'candidate1'],
                'logical_labels': [logical_pair['marker_a']['label'], logical_pair['marker_b']['label']],
                'observed_vector': best_forward['observed_vector'],
                'expected_vector_for_operator': best_forward['expected_vector'],
                'candidate_points': [candidates[0], candidates[1]],
                'all_operator_matches': forward_ranked,
            }
            alternative_ordering = {
                'observed_order': ['candidate1', 'candidate0'],
                'best_operator': best_reverse['operator'],
                'error_l1': best_reverse['error_l1'],
                'all_operator_matches': reverse_ranked,
            }
        else:
            best_match = {
                'operator': best_reverse['operator'],
                'error_l1': best_reverse['error_l1'],
                'exact': best_reverse['exact'],
                'observed_order': ['candidate1', 'candidate0'],
                'logical_labels': [logical_pair['marker_a']['label'], logical_pair['marker_b']['label']],
                'observed_vector': best_reverse['observed_vector'],
                'expected_vector_for_operator': best_reverse['expected_vector'],
                'candidate_points': [candidates[1], candidates[0]],
                'all_operator_matches': reverse_ranked,
            }
            alternative_ordering = {
                'observed_order': ['candidate0', 'candidate1'],
                'best_operator': best_forward['operator'],
                'error_l1': best_forward['error_l1'],
                'all_operator_matches': forward_ranked,
            }

        if best_match['error_l1'] != 0:
            warnings.append('orientation_nonexact_match')

    diagnosis = dict(module_diag)
    diagnosis.update({
        'diagnosis_type': 'asic_orientation',
        'logical_marker_pair': logical_pair,
        'candidate_scope': candidate_scope,
        'orientation': best_match if best_match is not None else {
            'operator': 'unresolved',
            'error_l1': None,
            'exact': False,
            'observed_order': [],
            'logical_labels': [],
            'observed_vector': None,
            'expected_vector_for_operator': None,
            'candidate_points': candidates,
            'all_operator_matches': [],
        },
        'alternative_ordering': alternative_ordering,
    })
    diagnosis['warnings'] = warnings
    return diagnosis


def _rank_quadrants_in_bbox(background_deviation, bbox):
    module = int(bbox['module'])
    arr = np.asarray(background_deviation[module, bbox['row0']:bbox['row1'], bbox['col0']:bbox['col1']], dtype=np.float64)
    if arr.shape != (176, 192):
        raise ValueError(f'expected block shape (176, 192), got {arr.shape}')

    ranked = []
    qrows = arr.shape[0] // 2
    qcols = arr.shape[1] // 2
    for qr in range(2):
        for qc in range(2):
            y0 = qr * qrows
            y1 = y0 + qrows
            x0 = qc * qcols
            x1 = x0 + qcols
            score = float(np.sum(arr[y0:y1, x0:x1], dtype=np.float64))
            ranked.append({
                'quadrant_row': int(qr),
                'quadrant_col': int(qc),
                'quadrant_label': BLOCK_LABELS[(qr, qc)],
                'score': score,
                'bbox_raw': {
                    'module': module,
                    'row0': int(bbox['row0'] + y0),
                    'row1': int(bbox['row0'] + y1),
                    'col0': int(bbox['col0'] + x0),
                    'col1': int(bbox['col0'] + x1),
                },
            })
    ranked.sort(key=lambda item: (-item['score'], item['quadrant_row'], item['quadrant_col']))
    return ranked


def _expected_quadrant_from_markers(markers):
    if not markers:
        return None
    rows = np.asarray([int(m['row']) for m in markers], dtype=np.float64)
    cols = np.asarray([int(m['col']) for m in markers], dtype=np.float64)
    mean_row = float(np.mean(rows))
    mean_col = float(np.mean(cols))
    qr = 0 if mean_row < 88.0 else 1
    qc = 0 if mean_col < 96.0 else 1
    return {
        'quadrant_row': int(qr),
        'quadrant_col': int(qc),
        'quadrant_label': BLOCK_LABELS[(qr, qc)],
        'mean_row': mean_row,
        'mean_col': mean_col,
    }


def _diagnose_quadrants(row, input_dir, min_total_score):
    module_diag = _diagnose_module_location(row, min_total_score)
    background_deviation = _load_run_array(input_dir, row, 'background_deviation.npy')
    bbox = module_diag['best_block']['bbox_raw']
    ranked_quadrants = _rank_quadrants_in_bbox(background_deviation, bbox)
    best_quadrant = ranked_quadrants[0]
    second_quadrant = ranked_quadrants[1] if len(ranked_quadrants) > 1 else None
    total_quadrant_score = float(sum(item['score'] for item in ranked_quadrants))
    best_quadrant_fraction = _safe_ratio(best_quadrant['score'], total_quadrant_score)
    best_quadrant_ratio = None if second_quadrant is None else _safe_ratio(best_quadrant['score'], second_quadrant['score'])
    best_quadrant_margin = None if second_quadrant is None else float(best_quadrant['score'] - second_quadrant['score'])

    expected_quadrant = _expected_quadrant_from_markers(module_diag.get('markers', []))
    warnings = list(module_diag.get('warnings', []))
    if best_quadrant['score'] <= 0:
        warnings.append('best_quadrant_nonpositive')
    if expected_quadrant is None:
        warnings.append('missing_expected_quadrant')

    diagnosis = dict(module_diag)
    diagnosis.update({
        'diagnosis_type': 'quadrants',
        'expected_quadrant': expected_quadrant,
        'ranked_quadrants_in_best_block': ranked_quadrants,
        'best_quadrant': {
            'module': int(bbox['module']),
            'quadrant_row': best_quadrant['quadrant_row'],
            'quadrant_col': best_quadrant['quadrant_col'],
            'quadrant_label': best_quadrant['quadrant_label'],
            'score': best_quadrant['score'],
            'fraction_of_block': best_quadrant_fraction,
            'ratio_to_second': best_quadrant_ratio,
            'margin_to_second': best_quadrant_margin,
            'bbox_raw': best_quadrant['bbox_raw'],
        },
    })
    diagnosis['warnings'] = warnings
    return diagnosis


def _rank_column_regions_in_bbox(background_deviation, bbox):
    module = int(bbox['module'])
    arr = np.asarray(background_deviation[module, bbox['row0']:bbox['row1'], bbox['col0']:bbox['col1']], dtype=np.float64)
    if arr.shape != (176, 192):
        raise ValueError(f'expected block shape (176, 192), got {arr.shape}')

    ranked = []
    band_width = arr.shape[1] // 4
    for region in range(4):
        x0 = region * band_width
        x1 = x0 + band_width
        score = float(np.sum(arr[:, x0:x1], dtype=np.float64))
        ranked.append({
            'region_index': int(region),
            'region_label': f'C{region}',
            'score': score,
            'bbox_raw': {
                'module': module,
                'row0': int(bbox['row0']),
                'row1': int(bbox['row1']),
                'col0': int(bbox['col0'] + x0),
                'col1': int(bbox['col0'] + x1),
            },
        })
    ranked.sort(key=lambda item: (-item['score'], item['region_index']))
    return ranked


def _expected_column_region_from_markers(markers):
    if not markers:
        return None
    cols = np.asarray([int(m['col']) for m in markers], dtype=np.float64)
    mean_col = float(np.mean(cols))
    region = min(3, max(0, int(mean_col // 48.0)))
    return {
        'region_index': int(region),
        'region_label': f'C{region}',
        'mean_col': mean_col,
    }


def _diagnose_column_regions(row, input_dir, min_total_score):
    module_diag = _diagnose_module_location(row, min_total_score)
    background_deviation = _load_run_array(input_dir, row, 'background_deviation.npy')
    bbox = module_diag['best_block']['bbox_raw']
    ranked_regions = _rank_column_regions_in_bbox(background_deviation, bbox)
    best_region = ranked_regions[0]
    second_region = ranked_regions[1] if len(ranked_regions) > 1 else None
    total_region_score = float(sum(item['score'] for item in ranked_regions))
    best_region_fraction = _safe_ratio(best_region['score'], total_region_score)
    best_region_ratio = None if second_region is None else _safe_ratio(best_region['score'], second_region['score'])
    best_region_margin = None if second_region is None else float(best_region['score'] - second_region['score'])

    expected_region = _expected_column_region_from_markers(module_diag.get('markers', []))
    warnings = list(module_diag.get('warnings', []))
    if best_region['score'] <= 0:
        warnings.append('best_column_region_nonpositive')
    if expected_region is None:
        warnings.append('missing_expected_column_region')

    diagnosis = dict(module_diag)
    diagnosis.update({
        'diagnosis_type': 'column_regions',
        'expected_column_region': expected_region,
        'ranked_column_regions_in_best_block': ranked_regions,
        'best_column_region': {
            'module': int(bbox['module']),
            'region_index': best_region['region_index'],
            'region_label': best_region['region_label'],
            'score': best_region['score'],
            'fraction_of_block': best_region_fraction,
            'ratio_to_second': best_region_ratio,
            'margin_to_second': best_region_margin,
            'bbox_raw': best_region['bbox_raw'],
        },
    })
    diagnosis['warnings'] = warnings
    return diagnosis


def _rank_row_regions_in_bbox(background_deviation, bbox):
    module = int(bbox['module'])
    arr = np.asarray(background_deviation[module, bbox['row0']:bbox['row1'], bbox['col0']:bbox['col1']], dtype=np.float64)
    if arr.shape != (176, 192):
        raise ValueError(f'expected block shape (176, 192), got {arr.shape}')

    ranked = []
    band_height = arr.shape[0] // 4
    for region in range(4):
        y0 = region * band_height
        y1 = y0 + band_height
        score = float(np.sum(arr[y0:y1, :], dtype=np.float64))
        ranked.append({
            'region_index': int(region),
            'region_label': f'R{region}',
            'score': score,
            'bbox_raw': {
                'module': module,
                'row0': int(bbox['row0'] + y0),
                'row1': int(bbox['row0'] + y1),
                'col0': int(bbox['col0']),
                'col1': int(bbox['col1']),
            },
        })
    ranked.sort(key=lambda item: (-item['score'], item['region_index']))
    return ranked


def _expected_row_region_from_markers(markers):
    if not markers:
        return None
    rows = np.asarray([int(m['row']) for m in markers], dtype=np.float64)
    mean_row = float(np.mean(rows))
    region = min(3, max(0, int(mean_row // 44.0)))
    return {
        'region_index': int(region),
        'region_label': f'R{region}',
        'mean_row': mean_row,
    }


def _diagnose_row_regions(row, input_dir, min_total_score):
    module_diag = _diagnose_module_location(row, min_total_score)
    background_deviation = _load_run_array(input_dir, row, 'background_deviation.npy')
    bbox = module_diag['best_block']['bbox_raw']
    ranked_regions = _rank_row_regions_in_bbox(background_deviation, bbox)
    best_region = ranked_regions[0]
    second_region = ranked_regions[1] if len(ranked_regions) > 1 else None
    total_region_score = float(sum(item['score'] for item in ranked_regions))
    best_region_fraction = _safe_ratio(best_region['score'], total_region_score)
    best_region_ratio = None if second_region is None else _safe_ratio(best_region['score'], second_region['score'])
    best_region_margin = None if second_region is None else float(best_region['score'] - second_region['score'])

    expected_region = _expected_row_region_from_markers(module_diag.get('markers', []))
    warnings = list(module_diag.get('warnings', []))
    if best_region['score'] <= 0:
        warnings.append('best_row_region_nonpositive')
    if expected_region is None:
        warnings.append('missing_expected_row_region')

    diagnosis = dict(module_diag)
    diagnosis.update({
        'diagnosis_type': 'row_regions',
        'expected_row_region': expected_region,
        'ranked_row_regions_in_best_block': ranked_regions,
        'best_row_region': {
            'module': int(bbox['module']),
            'region_index': best_region['region_index'],
            'region_label': best_region['region_label'],
            'score': best_region['score'],
            'fraction_of_block': best_region_fraction,
            'ratio_to_second': best_region_ratio,
            'margin_to_second': best_region_margin,
            'bbox_raw': best_region['bbox_raw'],
        },
    })
    diagnosis['warnings'] = warnings
    return diagnosis


def _diagnose_module_location(row, min_total_score):
    module_scores = np.asarray(row['module_scores'], dtype=np.float64)
    asic_scores = np.asarray(row['asic_scores'], dtype=np.float64)
    if module_scores.shape != (4,):
        raise ValueError(f"run {row['run']} has unexpected module_scores shape {module_scores.shape}")
    if asic_scores.shape != (4, 2, 2):
        raise ValueError(f"run {row['run']} has unexpected asic_scores shape {asic_scores.shape}")

    ranked_modules = _ranked_modules(module_scores)
    best_module = ranked_modules[0]
    second_module = ranked_modules[1] if len(ranked_modules) > 1 else None
    total_module_score = float(np.sum(module_scores))
    best_module_fraction = _safe_ratio(best_module['score'], total_module_score)
    best_module_ratio = None if second_module is None else _safe_ratio(best_module['score'], second_module['score'])
    best_module_margin = None if second_module is None else float(best_module['score'] - second_module['score'])

    best_module_index = best_module['module']
    ranked_blocks = _ranked_blocks(asic_scores[best_module_index])
    best_block = ranked_blocks[0]
    second_block = ranked_blocks[1] if len(ranked_blocks) > 1 else None
    total_block_score = float(np.sum(asic_scores[best_module_index]))
    best_block_fraction = _safe_ratio(best_block['score'], total_block_score)
    best_block_ratio = None if second_block is None else _safe_ratio(best_block['score'], second_block['score'])
    best_block_margin = None if second_block is None else float(best_block['score'] - second_block['score'])

    warnings = []
    if total_module_score <= min_total_score:
        warnings.append('total_module_score_at_or_below_threshold')
    if best_module['score'] <= 0:
        warnings.append('best_module_nonpositive')
    if best_block['score'] <= 0:
        warnings.append('best_block_nonpositive')

    return {
        'diagnosis_type': 'module_location',
        'run': int(row['run']),
        'pattern_index': int(row['pattern_index']),
        'pattern_label': row['pattern_label'],
        'test_name': row['test_name'],
        'marker_groups': row.get('marker_groups', []),
        'markers': row.get('markers', []),
        'events_used': int(row.get('events_used', 0)),
        'background_code': int(row.get('background_code', 0)),
        'total_module_score': total_module_score,
        'ranked_modules': ranked_modules,
        'best_module': {
            'module': best_module_index,
            'score': best_module['score'],
            'fraction_of_total': best_module_fraction,
            'ratio_to_second': best_module_ratio,
            'margin_to_second': best_module_margin,
        },
        'ranked_blocks_in_best_module': ranked_blocks,
        'best_block': {
            'module': best_module_index,
            'block_row': best_block['block_row'],
            'block_col': best_block['block_col'],
            'block_label': best_block['block_label'],
            'score': best_block['score'],
            'fraction_of_module': best_block_fraction,
            'ratio_to_second': best_block_ratio,
            'margin_to_second': best_block_margin,
            'bbox_raw': _coarse_bbox(best_module_index, best_block['block_row'], best_block['block_col']),
        },
        'top_candidates': row.get('top_candidates', []),
        'warnings': warnings,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_summary_md(path, diagnoses, diag_mode):
    if diag_mode == 'module_location':
        _write_module_location_summary_md(path, diagnoses)
    elif diag_mode == 'asic_orientation':
        _write_asic_orientation_summary_md(path, diagnoses)
    elif diag_mode == 'quadrants':
        _write_quadrants_summary_md(path, diagnoses)
    elif diag_mode == 'column_regions':
        _write_column_regions_summary_md(path, diagnoses)
    elif diag_mode == 'row_regions':
        _write_row_regions_summary_md(path, diagnoses)
    elif diag_mode == 'bank_marker_orientation':
        _write_bank_marker_orientation_summary_md(path, diagnoses)
    else:
        raise RuntimeError(f'unsupported summary mode {diag_mode!r}')


def _write_module_location_summary_md(path, diagnoses):
    lines = [
        '# Module Location Diagnosis',
        '',
        'Per-run coarse module/block diagnosis from extracted raw summaries.',
        '',
    ]
    for diag in diagnoses:
        best_module = diag['best_module']
        best_block = diag['best_block']
        lines.extend([
            f"## Run {diag['run']} Pattern {diag['pattern_index']:02d} {diag['pattern_label']}",
            '',
            f"- test_name: `{diag['test_name']}`",
            f"- marker_groups: `{diag['marker_groups']}`",
            f"- best_module: `{best_module['module']}` score={best_module['score']:.3f}",
            f"- best_module_fraction: `{best_module['fraction_of_total']}`",
            f"- best_block: `{best_block['block_label']}` in module `{best_block['module']}` score={best_block['score']:.3f}",
            f"- best_block_bbox_raw: `{best_block['bbox_raw']}`",
        ])
        if diag['warnings']:
            lines.append(f"- warnings: `{diag['warnings']}`")
        lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('\n'.join(lines))


def _write_asic_orientation_summary_md(path, diagnoses):
    lines = [
        '# ASIC Orientation Diagnosis',
        '',
        'Per-run coarse module/block diagnosis plus 2-point orientation classification.',
        '',
    ]
    for diag in diagnoses:
        best_module = diag['best_module']
        best_block = diag['best_block']
        orientation = diag['orientation']
        lines.extend([
            f"## Run {diag['run']} Pattern {diag['pattern_index']:02d} {diag['pattern_label']}",
            '',
            f"- test_name: `{diag['test_name']}`",
            f"- marker_groups: `{diag['marker_groups']}`",
            f"- best_module: `{best_module['module']}` score={best_module['score']:.3f}",
            f"- best_block: `{best_block['block_label']}` in module `{best_block['module']}` score={best_block['score']:.3f}",
            f"- candidate_scope: `{diag['candidate_scope']}`",
            f"- operator: `{orientation['operator']}`",
            f"- exact: `{orientation['exact']}`",
            f"- error_l1: `{orientation['error_l1']}`",
            f"- observed_vector: `{orientation['observed_vector']}`",
            f"- expected_vector_for_operator: `{orientation['expected_vector_for_operator']}`",
        ])
        if diag['warnings']:
            lines.append(f"- warnings: `{diag['warnings']}`")
        lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('\n'.join(lines))


def _write_quadrants_summary_md(path, diagnoses):
    lines = [
        '# Quadrant Diagnosis',
        '',
        'Per-run coarse module/block diagnosis plus quadrant ranking inside the winning block.',
        '',
    ]
    for diag in diagnoses:
        best_module = diag['best_module']
        best_block = diag['best_block']
        best_quadrant = diag['best_quadrant']
        expected = diag.get('expected_quadrant')
        lines.extend([
            f"## Run {diag['run']} Pattern {diag['pattern_index']:02d} {diag['pattern_label']}",
            '',
            f"- test_name: `{diag['test_name']}`",
            f"- marker_groups: `{diag['marker_groups']}`",
            f"- best_module: `{best_module['module']}` score={best_module['score']:.3f}",
            f"- best_block: `{best_block['block_label']}` in module `{best_block['module']}` score={best_block['score']:.3f}",
            f"- best_quadrant: `{best_quadrant['quadrant_label']}` score={best_quadrant['score']:.3f}",
            f"- best_quadrant_bbox_raw: `{best_quadrant['bbox_raw']}`",
        ])
        if expected is not None:
            lines.append(f"- expected_quadrant: `{expected['quadrant_label']}`")
        if diag['warnings']:
            lines.append(f"- warnings: `{diag['warnings']}`")
        lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('\n'.join(lines))


def _write_column_regions_summary_md(path, diagnoses):
    lines = [
        '# Column Region Diagnosis',
        '',
        'Per-run coarse module/block diagnosis plus 48-column-region ranking inside the winning block.',
        '',
    ]
    for diag in diagnoses:
        best_module = diag['best_module']
        best_block = diag['best_block']
        best_region = diag['best_column_region']
        expected = diag.get('expected_column_region')
        lines.extend([
            f"## Run {diag['run']} Pattern {diag['pattern_index']:02d} {diag['pattern_label']}",
            '',
            f"- test_name: `{diag['test_name']}`",
            f"- marker_groups: `{diag['marker_groups']}`",
            f"- best_module: `{best_module['module']}` score={best_module['score']:.3f}",
            f"- best_block: `{best_block['block_label']}` in module `{best_block['module']}` score={best_block['score']:.3f}",
            f"- best_column_region: `{best_region['region_label']}` score={best_region['score']:.3f}",
            f"- best_column_region_bbox_raw: `{best_region['bbox_raw']}`",
        ])
        if expected is not None:
            lines.append(f"- expected_column_region: `{expected['region_label']}`")
        if diag['warnings']:
            lines.append(f"- warnings: `{diag['warnings']}`")
        lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('\n'.join(lines))


def _write_row_regions_summary_md(path, diagnoses):
    lines = [
        '# Row Region Diagnosis',
        '',
        'Per-run coarse module/block diagnosis plus row-band ranking inside the winning block.',
        '',
    ]
    for diag in diagnoses:
        best_module = diag['best_module']
        best_block = diag['best_block']
        best_region = diag['best_row_region']
        expected = diag.get('expected_row_region')
        lines.extend([
            f"## Run {diag['run']} Pattern {diag['pattern_index']:02d} {diag['pattern_label']}",
            '',
            f"- test_name: `{diag['test_name']}`",
            f"- marker_groups: `{diag['marker_groups']}`",
            f"- best_module: `{best_module['module']}` score={best_module['score']:.3f}",
            f"- best_block: `{best_block['block_label']}` in module `{best_block['module']}` score={best_block['score']:.3f}",
            f"- best_row_region: `{best_region['region_label']}` score={best_region['score']:.3f}",
            f"- best_row_region_bbox_raw: `{best_region['bbox_raw']}`",
        ])
        if expected is not None:
            lines.append(f"- expected_row_region: `{expected['region_label']}`")
        if diag['warnings']:
            lines.append(f"- warnings: `{diag['warnings']}`")
        lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('\n'.join(lines))


def _write_bank_marker_orientation_summary_md(path, diagnoses):
    lines = [
        '# Bank Marker Orientation Diagnosis',
        '',
        'Per-run coarse module/block diagnosis plus sparse in-bank 2-point orientation classification.',
        '',
    ]
    for diag in diagnoses:
        best_module = diag['best_module']
        best_block = diag['best_block']
        orient = diag['bank_marker_orientation']
        lines.extend([
            f"## Run {diag['run']} Pattern {diag['pattern_index']:02d} {diag['pattern_label']}",
            '',
            f"- test_name: `{diag['test_name']}`",
            f"- marker_groups: `{diag['marker_groups']}`",
            f"- best_module: `{best_module['module']}` score={best_module['score']:.3f}",
            f"- best_block: `{best_block['block_label']}` in module `{best_block['module']}` score={best_block['score']:.3f}",
            f"- coordinate_mode: `{diag['coordinate_mode']}`",
            f"- bank_index: `{diag['bank_index']}`",
            f"- bank_bbox_raw: `{diag['bank_bbox_raw']}`",
            f"- operator: `{orient['operator']}`",
            f"- exact: `{orient['exact']}`",
            f"- error_l1: `{orient['error_l1']}`",
            f"- observed_vector: `{orient['observed_vector']}`",
            f"- expected_vector_for_operator: `{orient['expected_vector_for_operator']}`",
        ])
        if diag['warnings']:
            lines.append(f"- warnings: `{diag['warnings']}`")
        lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('\n'.join(lines))


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format='[%(levelname).1s] %(message)s',
    )

    input_dir = Path(args.input_dir).expanduser().resolve()
    run_summaries_path = input_dir / 'run_summaries.json'
    if not run_summaries_path.exists():
        raise RuntimeError(f'missing run summaries file: {run_summaries_path}')
    if args.diag not in ('module_location', 'asic_orientation', 'quadrants', 'column_regions', 'row_regions', 'bank_marker_orientation'):
        raise RuntimeError(
            f"unsupported --diag {args.diag!r}; supported: module_location, asic_orientation, quadrants, column_regions, row_regions, bank_marker_orientation"
        )

    run_summaries = _load_json(run_summaries_path)
    run_selector = args.run if args.run is not None else args.runs
    rows = _selected_rows(run_summaries, run_selector, args.patterns)
    if not rows:
        raise RuntimeError('no runs selected for diagnosis')

    diagnoses = []
    outdir = input_dir / 'diagnosis'
    for row in rows:
        if args.diag == 'module_location':
            diag = _diagnose_module_location(row, args.min_total_score)
            per_run_name = 'module_location.json'
            combined_name = 'module_location_all.json'
            summary_name = 'module_location_summary.md'
            log_msg = 'diagnosed run=%d pattern=%02d best_module=%d best_block=%s'
            log_args = (
                diag['run'], diag['pattern_index'],
                diag['best_module']['module'], diag['best_block']['block_label'],
            )
        elif args.diag == 'asic_orientation':
            diag = _diagnose_asic_orientation(row, args.min_total_score)
            per_run_name = 'asic_orientation.json'
            combined_name = 'asic_orientation_all.json'
            summary_name = 'asic_orientation_summary.md'
            log_msg = 'diagnosed run=%d pattern=%02d best_module=%d best_block=%s operator=%s'
            log_args = (
                diag['run'], diag['pattern_index'],
                diag['best_module']['module'], diag['best_block']['block_label'],
                diag['orientation']['operator'],
            )
        elif args.diag == 'bank_marker_orientation':
            diag = _diagnose_bank_marker_orientation(
                row,
                input_dir,
                args.min_total_score,
                args.coordinate_mode,
                args.bank_index,
                args.peak_exclusion_radius,
            )
            per_run_name = 'bank_marker_orientation.json'
            combined_name = 'bank_marker_orientation_all.json'
            summary_name = 'bank_marker_orientation_summary.md'
            log_msg = 'diagnosed run=%d pattern=%02d best_module=%d best_block=%s bank=%d operator=%s'
            log_args = (
                diag['run'], diag['pattern_index'],
                diag['best_module']['module'], diag['best_block']['block_label'],
                diag['bank_index'], diag['bank_marker_orientation']['operator'],
            )
        else:
            if args.diag == 'quadrants':
                diag = _diagnose_quadrants(row, input_dir, args.min_total_score)
                per_run_name = 'quadrants.json'
                combined_name = 'quadrants_all.json'
                summary_name = 'quadrants_summary.md'
                log_msg = 'diagnosed run=%d pattern=%02d best_module=%d best_block=%s best_quadrant=%s'
                log_args = (
                    diag['run'], diag['pattern_index'],
                    diag['best_module']['module'], diag['best_block']['block_label'],
                    diag['best_quadrant']['quadrant_label'],
                )
            else:
                if args.diag == 'column_regions':
                    diag = _diagnose_column_regions(row, input_dir, args.min_total_score)
                    per_run_name = 'column_regions.json'
                    combined_name = 'column_regions_all.json'
                    summary_name = 'column_regions_summary.md'
                    log_msg = 'diagnosed run=%d pattern=%02d best_module=%d best_block=%s best_column_region=%s'
                    log_args = (
                        diag['run'], diag['pattern_index'],
                        diag['best_module']['module'], diag['best_block']['block_label'],
                        diag['best_column_region']['region_label'],
                    )
                else:
                    diag = _diagnose_row_regions(row, input_dir, args.min_total_score)
                    per_run_name = 'row_regions.json'
                    combined_name = 'row_regions_all.json'
                    summary_name = 'row_regions_summary.md'
                    log_msg = 'diagnosed run=%d pattern=%02d best_module=%d best_block=%s best_row_region=%s'
                    log_args = (
                        diag['run'], diag['pattern_index'],
                        diag['best_module']['module'], diag['best_block']['block_label'],
                        diag['best_row_region']['region_label'],
                    )
        diagnoses.append(diag)
        run_dir = outdir / ('run%04d_pattern%02d' % (diag['run'], diag['pattern_index']))
        _write_json(run_dir / per_run_name, diag)
        logger.info(log_msg, *log_args)

    _write_json(outdir / combined_name, diagnoses)
    _write_summary_md(outdir / summary_name, diagnoses, args.diag)
    logger.info('wrote %d %s diagnosis records to %s', len(diagnoses), args.diag, str(outdir))


if __name__ == '__main__':
    main()

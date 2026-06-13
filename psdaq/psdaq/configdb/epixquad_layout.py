"""Shared raw-layout geometry for epixquad config tools.

This module intentionally contains only lightweight constants so scripts can
share the same detector tiling without importing hardware/runtime modules.
"""

ASIC_ROWS = 176
ASIC_COLS = 192
ASIC_COUNT = 16
MODULE_COUNT = 4
MODULE_ROWS = ASIC_ROWS * 2
MODULE_COLS = ASIC_COLS * 2
RAW_SHAPE = (MODULE_COUNT, MODULE_ROWS, MODULE_COLS)

RAW_ASIC_LAYOUT = (
    {'slot': 0, 'row_slice': (176, 352), 'col_slice': (192, 384), 'operator': 'identity'},
    {'slot': 1, 'row_slice': (0, 176),   'col_slice': (192, 384), 'operator': 'rot180'},
    {'slot': 2, 'row_slice': (0, 176),   'col_slice': (0, 192),   'operator': 'rot180'},
    {'slot': 3, 'row_slice': (176, 352), 'col_slice': (0, 192),   'operator': 'identity'},
)


def raw_detector_view(array):
    """Tile raw epixquad segments into the shared 2x2 detector-panel view."""
    import numpy as np

    array = np.asarray(array)
    if array.shape != RAW_SHAPE:
        raise ValueError(f'raw_detector_view expects shape {RAW_SHAPE}, got {array.shape}')
    return np.vstack([
        np.hstack([array[3], array[2]]),
        np.hstack([array[1], array[0]]),
    ])

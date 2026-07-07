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
DETECTOR_VIEW_SHAPE = (MODULE_ROWS * 2, MODULE_COLS * 2)
EPIXVIEWER_DECODED_SHAPE = (712, 768)

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


def detector_view_to_raw(array):
    """Convert the shared 2x2 detector-panel view back to raw DAQ segment order."""
    import numpy as np

    array = np.asarray(array)
    if array.shape != DETECTOR_VIEW_SHAPE:
        raise ValueError(
            f'detector_view_to_raw expects shape {DETECTOR_VIEW_SHAPE}, got {array.shape}'
        )
    raw = np.empty(RAW_SHAPE, dtype=array.dtype)
    raw[3] = array[:MODULE_ROWS, :MODULE_COLS]
    raw[2] = array[:MODULE_ROWS, MODULE_COLS:]
    raw[1] = array[MODULE_ROWS:, :MODULE_COLS]
    raw[0] = array[MODULE_ROWS:, MODULE_COLS:]
    return raw


def epixviewer_decoded_to_daq_raw(decoded):
    """Convert ePixViewer ePixQuad StreamWriter images to DAQ/psana raw layout.

    Rogue StreamWriter files are decoded through ePixViewer as a single
    ``(712, 768)`` image.  The DAQ writes XTC data as four raw panels with
    shape ``(4, 352, 384)``.  This conversion mirrors the ordering in
    ``psdaq/drp/EpixQuad.cc`` so downstream checks see the same coordinates as
    ``det.raw.raw(evt)``.
    """
    import numpy as np

    decoded = np.asarray(decoded)
    if decoded.shape != EPIXVIEWER_DECODED_SHAPE:
        raise ValueError(
            f'epixviewer_decoded_to_daq_raw expects shape {EPIXVIEWER_DECODED_SHAPE}, '
            f'got {decoded.shape}'
        )

    raw = np.empty(RAW_SHAPE, dtype=decoded.dtype)
    raw[0] = decoded[2:354, 0:384][::-1, ::-1]
    raw[1] = decoded[2:354, 384:768][::-1, ::-1]
    raw[2] = decoded[358:710, 0:384][::-1, ::-1]
    raw[3] = decoded[358:710, 384:768][::-1, ::-1]
    return raw


def daq_raw_to_epixviewer_decoded(raw, fill_value=0):
    """Convert DAQ/psana raw layout to ePixViewer display layout.

    The ePixViewer-only rows 0, 1, 354, 355, 356, 357, 710, and 711 are not
    part of ``det.raw.raw(evt)``.  They are filled with ``fill_value``.
    """
    import numpy as np

    raw = np.asarray(raw)
    if raw.shape != RAW_SHAPE:
        raise ValueError(f'daq_raw_to_epixviewer_decoded expects shape {RAW_SHAPE}, got {raw.shape}')

    decoded = np.full(EPIXVIEWER_DECODED_SHAPE, fill_value, dtype=raw.dtype)
    decoded[2:354, 0:384] = raw[0][::-1, ::-1]
    decoded[2:354, 384:768] = raw[1][::-1, ::-1]
    decoded[358:710, 0:384] = raw[2][::-1, ::-1]
    decoded[358:710, 384:768] = raw[3][::-1, ::-1]
    return decoded

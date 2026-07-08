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
DAQ_RAW_FRAME_WORDS = MODULE_COUNT * MODULE_ROWS * MODULE_COLS
DAQ_RAW_FRAME_BYTES = DAQ_RAW_FRAME_WORDS * 2

# The Rogue StreamWriter payload contains one or more 40-byte headers before
# the ePixQuad image words.  The common full-frame payload is 1095248 bytes
# with one 20-word header.  Some frames carry one extra 40-byte header and are
# 1095288 bytes long.
ROGUE_FULL_FRAME_BASE_BYTES = 1095248
ROGUE_IMAGE_BASE_SKIP_WORDS = 20

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

    This compatibility helper is for already-decoded ePixViewer display images
    with shape ``(712, 768)``.  For Rogue StreamWriter payloads, use
    ``rogue_payload_to_daq_raw`` so the record header offset is handled before
    the row/column mapping.
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


def rogue_image_skip_words(payload_nbytes):
    """Return the image-data start offset for an ePixQuad Rogue payload.

    The DAQ DRP code starts image extraction at the first camera image word, not
    at the ePixViewer display offset.  StreamWriter records seen on rdsrv421
    use a 20-word image offset for 1095248-byte frames and a 40-word offset for
    the 1095288-byte frames that include one extra 40-byte header.
    """
    extra = int(payload_nbytes) - ROGUE_FULL_FRAME_BASE_BYTES
    if extra <= 0:
        return ROGUE_IMAGE_BASE_SKIP_WORDS
    if extra % 2:
        raise ValueError(f"odd ePixQuad payload-size delta: {payload_nbytes}")
    return ROGUE_IMAGE_BASE_SKIP_WORDS + extra // 2


def rogue_payload_to_daq_raw(payload, bit_mask=0xFFFF):
    """Decode an ePixQuad Rogue StreamWriter image payload as DAQ raw panels.

    This mirrors ``psdaq/drp/EpixQuad.cc``: starting from the camera image words,
    each 384-word row is copied into the destination panel in reverse column
    order.  It avoids ePixViewer's display-image path, whose fixed 32-byte
    header skip is not correct for these StreamWriter records.
    """
    import numpy as np

    payload_nbytes = len(payload)
    skip_words = rogue_image_skip_words(payload_nbytes)
    required_nbytes = (skip_words * 2) + DAQ_RAW_FRAME_BYTES
    if payload_nbytes < required_nbytes:
        raise ValueError(
            f"payload has {payload_nbytes} bytes, need at least {required_nbytes} "
            f"for skip_words={skip_words}"
        )

    words = np.frombuffer(payload, dtype="<u2")
    raw = np.empty(RAW_SHAPE, dtype=np.uint16)
    index = skip_words
    for i in range(ASIC_ROWS):
        down_row = ASIC_ROWS + i
        up_row = ASIC_ROWS - i - 1
        for segment, row in (
            (2, up_row),
            (3, up_row),
            (2, down_row),
            (3, down_row),
            (0, up_row),
            (1, up_row),
            (0, down_row),
            (1, down_row),
        ):
            raw[segment, row] = words[index:index + MODULE_COLS][::-1]
            index += MODULE_COLS

    if bit_mask is not None:
        raw &= np.uint16(bit_mask)
    return raw, skip_words


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

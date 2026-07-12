"""Unit coverage for GPU stream segment ordering."""

from psana.gpu.gpu_calib import _segment_ids_in_l1_order


class _EventDgram:
    # Deliberately non-sorted: this models insertion by dgram.cc while walking
    # ShapesData children in an L1Accept payload.
    jungfrau = {
        17: object(),
        13: object(),
        9: object(),
        5: object(),
        29: object(),
        25: object(),
        21: object(),
    }


def test_segment_ids_preserve_l1_child_order():
    assert _segment_ids_in_l1_order(_EventDgram(), "jungfrau") == [
        17,
        13,
        9,
        5,
        29,
        25,
        21,
    ]


def test_missing_detector_returns_empty_order():
    assert _segment_ids_in_l1_order(object(), "jungfrau") == []

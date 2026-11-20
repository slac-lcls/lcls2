"""
Interfaces for ROIBin algorithm output from the GPU DRP (DrpAlgos).

ROIBin (like all algorithms currently) writes the output as uint8_t bytes under the
`reduced` field name.

Algorithm parameters are available in the configure transition. For a single DRP
algorithm, they will be present under the `drp_alg0` entry. (The number at the end
can in principle increase, if multiple algorithms were to be run in a row.)

Utilities for decoding/geometry that are currently used by the algorithm for
configuration are available from the Python `drpalgos` package, which comes from
the wider `DRP-algos` project.

The v0.1.0/v0.2.0 interface is currently decoded as follows by this psana interface:
- det.ROIBin.binned - A list of binned ROIs, or None
- det.ROIBin.header - The ROIBin header information (for time binning esp.)
                      Contains time-bin index, frame count, and flags. From v0.2.0 it
                      also has a presence bit-mask (for vetoed events)
- det.ROIBin.image - The ROI image. These are just drawn side-by-side if there are
                     multiple segments/panels
- det.ROIBin.rois - A list of the per-panel ROI bounds (written half-open)
- det.ROIBin.params - The configured parameters.

Classes:
    epixuhr3x2hw_ROIBin_0_1_0: ROIBin 0.1.0 output of the ePix UHR 3x2.
    epixuhr3x2_ROIBin_0_1_0: ROIBin 0.1.0 output of the ePix UHR 3x2.
    epixuhr_ROIBin_0_1_0: For use with the tdet emulation.
    epixuhremu_ROIBin_0_1_0: For use with the tdet emulation.

    epixuhr3x2hw_ROIBin_0_2_0: ROIBin 0.2.0 output of the ePix UHR 3x2.
    epixuhr3x2_ROIBin_0_2_0: ROIBin 0.2.0 output of the ePix UHR 3x2.
    epixuhr_ROIBin_0_2_0: For use with the tdet emulation.
    epixuhremu_ROIBin_0_2_0: For use with the tdet emulation.
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
from amitypes import Array2d

from psana.detector.detector_impl import DetectorImpl

logger: logging.Logger = logging.getLogger(__name__)

_drpalgos_error: Optional[ImportError] = None
try:
    import drpalgos.readers.roibin as rbr
except ImportError as err:
    _drpalgos_error = err


def _seg_rois(params: Dict[str, Any]) -> List[Tuple[Sequence[int], Sequence[int]]]:
    """(roi_start, roi_shape) for each ROI of one segment, in output order."""
    if "num_rois" in params:
        num_rois: int = int(params["num_rois"])
        return list(zip(params["roi_start"][:num_rois], params["roi_shape"][:num_rois]))
    return [(params["roi_start"], params["roi_shape"])]


class _ROIBinBase(DetectorImpl):
    """Base implementaiton for ROIBin decoding.

    Should be detector-agnostic, just need to sub-class as needed for different
    detTypes.
    """

    _alg_name: ClassVar[str] = "ROIBin"
    _version: ClassVar[Tuple[int, int, int]] = (0, 0, 0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _drpalgos_error is not None:
            logger.warning(
                f"{type(self).__name__}: drpalgos unavailable, "
                f"ROIBin data cannot be decoded ({_drpalgos_error})"
            )

    def _reset(self, *args, **kwargs) -> None:
        super()._reset(*args, **kwargs)
        self._params_cache: Optional[Dict[int, Dict[str, Any]]] = None
        self._geom_cache: Any = None

    def _require(self) -> None:
        if _drpalgos_error is not None:
            raise ImportError(
                f"{type(self).__name__} needs the drpalgos package: {_drpalgos_error}"
            ) from _drpalgos_error

    def _find_params(self) -> Dict[int, Dict[str, Any]]:
        """Collect each segment's parameters from the configure transition."""
        found: Dict[int, Dict[str, Any]] = {}
        for dgram in self._configs:
            for name in (self._det_name, self._det_name + "hw"):
                segs: Any = getattr(dgram, name, None)
                if not isinstance(segs, dict):
                    continue
                for seg, seg_cfg in segs.items():
                    for drp_class in vars(seg_cfg).values():
                        drp_alg: Any = getattr(drp_class, "drp_alg0", None)
                        if getattr(drp_alg, "name", None) == self._alg_name:
                            found.setdefault(seg, rbr.seg_params_from_config(drp_alg))

        missing: List[int] = sorted(set(self._sorted_segment_inds) - set(found))
        if missing:
            logger.warning(
                f"{self._det_name}: no {self._alg_name} parameters for segments {missing}"
            )

        return found

    def params(self, evt=None) -> Dict[int, Dict[str, Any]]:
        """ROIBin parameters per segment, from CONFIGURE."""
        self._require()
        if self._params_cache is None:
            self._params_cache = self._find_params()
        return self._params_cache

    def _geometry(self) -> Any:
        """The side-by-side goemetry used by the control GUI.

        NOTE: This is a simplified geometry not a full replica of the psana geometry.
              It is pixel-accurate, and does not embed into a 2D image with per-panel
              positioning.
        """
        if self._geom_cache is None:
            self._geom_cache = rbr.detector_geometry(self._dettype, self.params())

        return self._geom_cache

    def _decoded(self, evt) -> Optional[Dict[int, Any]]:
        """Decode the raw bytes per panel."""
        self._require()
        segs: Any = self._segments(evt)
        if segs is None:
            return None

        params: Dict[int, Dict[str, Any]] = self.params()
        outputs: Dict[int, Any] = {}
        for seg, container in segs.items():
            if seg in params:
                out = rbr.decode(container.reduced, params[seg], self._version)
                if out is not None:
                    outputs[seg] = out

        return outputs or None

    def header(self, evt) -> Optional[Dict[int, Any]]:
        """The header written by ROIBin.

        The header includes the time-bin index and frame counts, alogn with flags.

        Beginning with v0.2.0, the header also has presence bit-masks, to indicate
        which frames in the bin were included/vetoed.

        This is needed for time-bin interpretation. The output is None if time
        binning was not configured.
        """
        outputs = self._decoded(evt)
        return None if outputs is None else {s: o.header for s, o in outputs.items()}

    def binned(self, evt) -> Optional[Dict[int, List[np.ndarray]]]:
        """A list of binned ROIs for each segment.

        NOTE: It returns None for events without output, e.g. intermediate frames
              for when time binning is enabled. (Time bins are only written on last
              included event)
        """
        outputs = self._decoded(evt)
        if outputs is None:
            return None
        return {
            seg: [roi[0] if roi.shape[0] == 1 else roi for roi in output.data]
            for seg, output in outputs.items()
        }

    def image(self, evt, fill: float = np.nan) -> Array2d:
        """The binned ROIs laid out side-by-side for display.

        NOTE: This places them using the simplified geometry used by the control GUI.
        """
        outputs = self._decoded(evt)
        if outputs is None:
            return None
        return rbr.detector_image(outputs, self.params(), self._geometry(), fill)

    def rois(self, evt=None) -> List[List[int]]:
        """A list of the per-segment ROI bounds.

        NOTE: The ROIs bounds are written half open: [row0, col0, row1, col1)
        """
        self._require()

        geom: Any = self._geometry()
        bounds: List[List[int]] = []
        for seg in sorted(self.params()):
            for (_, row0, col0), (_, height, width) in _seg_rois(self.params()[seg]):
                if height and width:
                    rect = (
                        int(row0),
                        int(col0),
                        int(row0) + int(height),
                        int(col0) + int(width),
                    )
                    bounds.append(list(geom.to_detector(seg, rect)))

        return bounds


class _ROIBinV010(_ROIBinBase):
    _version = (0, 1, 0)


class _ROIBinV020(_ROIBinBase):
    _version = (0, 2, 0)


class epixuhr3x2hw_ROIBin_0_1_0(_ROIBinV010):
    """ROIBin 0.1.0 output of the ePix UHR 3x2 GPU DRP."""

    ...


class epixuhr3x2_ROIBin_0_1_0(_ROIBinV010):
    """ROIBin 0.1.0 output of the ePix UHR 3x2 GPU DRP."""

    ...


class epixuhr_ROIBin_0_1_0(_ROIBinV010):
    """ROIBin 0.1.0 output of the (fakecam UHR emulator) GPU DRP."""

    ...


class epixuhremu_ROIBin_0_1_0(_ROIBinV010):
    """ROIBin 0.1.0 output of the (fakecam UHR emulator) GPU DRP."""

    ...


class epixuhr3x2hw_ROIBin_0_2_0(_ROIBinV020):
    """ROIBin 0.2.0 output of the ePix UHR 3x2 GPU DRP."""

    ...


class epixuhr3x2_ROIBin_0_2_0(_ROIBinV020):
    """ROIBin 0.2.0 output of the ePix UHR 3x2 GPU DRP."""

    ...


class epixuhr_ROIBin_0_2_0(_ROIBinV020):
    """ROIBin 0.2.0 output of the (fakecam UHR emulator) GPU DRP."""

    ...


class epixuhremu_ROIBin_0_2_0(_ROIBinV020):
    """ROIBin 0.2.0 output of the (fakecam UHR emulator) GPU DRP."""

    ...

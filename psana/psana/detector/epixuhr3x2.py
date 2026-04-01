import logging
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from amitypes import Array3d

import psana.container
import psana.detector.epix_base as eb
from psana.detector.detector_impl import DetectorImpl

logger: logging.Logger = logging.getLogger(__name__)


class epixuhr3x2hw_config_0_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class epixuhr3x2_config_0_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class epixuhr3x2_raw_0_1_0(eb.epix_base):
    def __init__(self, *args, **kwargs):
        eb.epix_base.__init__(self, *args, **kwargs)

    def raw(self, evt) -> Array3d:
        r"""Return the raw unpacked data.

        The ePixUHR3x2, when not using "gain expansion" transmits the data as 12
        bits, in a 16 bit integer. The layout of this data is:

                           G D D D D D D D D D D D U U U U
                           | \___________________/ \_____/
                          /            |              |
                     Gain bit   11 bits of data  Unused bits

        Given the above representation, the data is not "packed" in the traditional
        sense. For space saving, the DAQ WILL pack the data, removing the unused bits.

        The data is then stored in the format:
                         (NumAsics, NumPackedInts)
        for each of the panels participating the DAQ.

        Each detector panel has 6 asics, each with shape (168, 192). These are
        arranged in the following format:

                              A1   |   A3   |   A5
                           --------+--------+--------
                              A0   |   A2   |   A4

        The raw retrieval function will deal with unpacking and reshaping. The final
        output will be of shape:
                         (NumPanels, 336, 576)
        """
        if evt is None:
            return None

        segs: Optional[dict[int, Any]] = self._segments(evt)
        if segs is None:
            return None
        nsegs: int = len(segs)
        n_asic_rows: int = 168
        n_asic_cols: int = 192
        # Final output will be as described above. (NPanels, 2*NRows, 3*NCols)
        arr: npt.NDArray[np.uint16] = np.zeros(
            (nsegs, n_asic_rows * 2, n_asic_cols * 3), dtype=np.uint16
        )

        # E.g. for 2 panels, we will have (2, 6, npixels)
        # We will reshape it into (2, 192*2, 168*3)
        # NOTE: This loop currently assumes that seg_idx starts at 0.
        #       If not, then the enumerate segs_seen needs to be used for indexing
        #       the output array. Or something needs to be decided in terms of how
        #       to handle missing data in the event that DRP segment numbers don't start
        #       at 0.
        for _, seg_idx in enumerate(segs):
            seg: psana.container.Container = segs[seg_idx]
            unpacked: npt.NDArray[np.uint16] = self._unpackData(seg.raw)

            # As a final step, reshape the data into a physical shape.
            blocked_asics: npt.NDArray[np.uint16] = unpacked.reshape(
                2, 3, n_asic_rows, n_asic_cols
            )
            arranged: npt.NDArray[np.uint16] = blocked_asics.transpose(
                0, 2, 1, 3
            ).reshape(n_asic_rows * 2, n_asic_cols * 3)

            arr[seg_idx] = arranged

        return arr

    def _unpackData(
        self, packed_data: npt.NDArray[np.uint16]
    ) -> npt.NDArray[np.uint16]:
        """Given a single panel's packed representation, unpack into 6*192*168 pixels.

        Args:
            packed_data (npt.NDArray[np.uint16]): The packed panel representation
                as provided by the DAQ.

        Returns:
            unpacked_data (npt.NDArray[np.uint16]): An unpacked representation.
                This function does not reshape the data and gives a flat representation
                of the 6 asics.
        """
        raw_bytes_data: npt.NDArray[np.uint8] = packed_data.view(np.uint8)

        num_pixels: int = (raw_bytes_data.size // 3) * 2

        # Assert that we can decompose properly
        assert raw_bytes_data.size % 3 == 0
        assert num_pixels % 2 == 0

        # Reshape into triplets of 3 bytes for our bitwise ops to unpack
        triplets: npt.NDArray[np.uint8] = raw_bytes_data.reshape(-1, 3)
        b0 = triplets[:, 0].astype(np.uint16)  # Cast now to avoid overflow
        b1 = triplets[:, 1].astype(np.uint16)
        b2 = triplets[:, 2].astype(np.uint16)

        out: npt.NDArray[np.uint16] = np.empty(num_pixels, dtype=np.uint16)

        out[0::2] = (b0 << 4) | (b1 >> 4)
        out[1::2] = ((b1 & 0x0F) << 8) | b2

        return out

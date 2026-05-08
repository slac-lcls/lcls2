import logging
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from amitypes import Array2d, Array3d
from time import time

import psana.container
import psana.detector.epix_base as eb
from psana.detector.detector_impl import DetectorImpl

import psana.detector.NDArrUtils as ndu
cond_msg = eb.ue.cond_msg

SEGMENT_SHAPE = (336, 576)
#ASIC_SHAPE    = (168, 192)

logger: logging.Logger = logging.getLogger(__name__)

def rot180(arr2d):
    return np.flipud(np.fliplr(arr2d))

def bit_opers_2x3(a):
    """bit massaging of panel data"""
    gbit = a & 0o100000                # save array with gain bit in 16-th position
    a = np.right_shift(a & 0o77777, 4) # mask 15 lower bits of data and move them 4 bits right
    a = np.bitwise_or(a, gbit)         # set the gain bit
    return a

def reshape_6x32256_to_6x168x192(a, shape_out=(6, 168, 192)):
    """returns array shaped as (2*3, 168, 192) from raw shape (6, 32256)"""
    a.shape = shape_out # (2,3,) + shape_asic
    return a

def stack_2x3_asics(a):
    """stacks asic from input array of shape (6, 168, 192) into 2d segment array,
       returns array shaped as (336, 576)=(2*168, 3*192)
       Dawood's numeration from
       https://confluence.slac.stanford.edu/spaces/ppareg/pages/655311578/ASIC+layout+and+Carrier+orientation

               *|       *|       * <- (0,0) pixel of (168, 192) == (rows, cols)
           A0   |   A1   |   A2
        --------+--------+--------
           A3   |   A4   |   A5
        *       |*       |*

    """
    return np.vstack((np.hstack((np.fliplr(a[0,:]), np.fliplr(a[1,:]), np.fliplr(a[2,:]))),\
                      np.hstack((np.flipud(a[3,:]), np.flipud(a[4,:]), np.flipud(a[5,:])))))
    #return np.vstack((np.hstack((rot180(a[1,:]), rot180(a[3,:]), rot180(a[5,:])))),\
    #                  np.hstack((       a[0,:],         a[2,:],         a[4,:]))))

class epixuhr3x2hw_config_0_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class epixuhr3x2_config_0_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class epixuhr3x2_raw_0_1_0(eb.epix_base):
    def __init__(self, *args, **kwargs):
        eb.epix_base.__init__(self, *args, **kwargs)
        self._gain_modes = ('FHG', 'FMG', 'FLG1', 'FLG2', 'AHLG1', 'AHLG2', 'AMLG1', 'AMLG2')
        self._counter_image = 0

    def raw(self, evt, sh_seg=(336,576)) -> Array3d:
        if cond_msg(evt is None, msg='evt is None - return None', output_meth=logger.warning):
            return None
        segs = self._segments(evt) # {0: <psana.container.Container object at 0x7f9cd51e0bd0>}
        segnums = self._segment_numbers # for now [0,]
        maxsegnum = max(segnums)
        out = np.zeros((maxsegnum+1,)+sh_seg, dtype=np.uint16)
        for iseg, nseg in enumerate(segnums):
            raw_asics = segs[iseg].raw # shape:(6, 32256)
            arr2 = bit_opers_2x3(raw_asics)
            asics = reshape_6x32256_to_6x168x192(arr2) # (4, 168, 192)
            arrseg = stack_2x3_asics(asics) # (336, 576)
            out[nseg,:] = arrseg # save panel in the output array
        return out

    def image(self, evt, **kwargs) -> Array2d: # see in areadetector.py
        if cond_msg(evt is None, msg='evt is None - return None', output_meth=logger.warning):
            return None
        self._counter_image += 1
        if self._counter_image < 3:
            logger.warning('TBD TEMPORARY det.raw.image returns 0-th panel: det.raw.image(evt) = det.raw.raw(evt)[0,:]')
        return self.raw(evt)[0,:]

    def _cbits_config_segment(self, cob):
        """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object, where self=det.raw
           returns segment gain control bits # shape=(336, 576)
        """
        logger.debug('XXX dir(cob): %s' % str(dir(cob))) #'gainAsic', 'gainCSVAsic']
        gasic = cob.gainAsic            # [56 56 56 56 56 56]
        cbits = cob.gainCSVAsic.copy()  # shape:(6, 32256)
        logger.debug('  XXX cob.gainAsic: %s' % str(gasic))
        logger.debug(ndu.info_ndarr(cbits, '  XXX cob.gainCSVAsic', last=10))
        cbits.shape = (6, 168, 192) # reshape_6x32256_to_6x168x192(cbits)
        for i, g in enumerate(gasic):
            if g > 0: cbits[i,:] = g
        cbits = stack_2x3_asics(cbits) # (336, 576)
        logger.info(ndu.info_ndarr(cbits, 'segment cbits', last=10))
        return cbits
       # return eb.cbits_config_epix10ka(cob, shape=(352, 384)) # in epix10ka.py



## LEGACY STUFF from Gabriel
#    it did not work... due to reshaping, assembling of asics etc.

#    def _raw_v0(self, evt) -> Array3d:
#        r"""Return the raw unpacked data.
#
#        The ePixUHR3x2, when not using "gain expansion" transmits the data as 12
#        bits, in a 16 bit integer. The layout of this data is:
#
#                           G D D D D D D D D D D D U U U U
#                           | \___________________/ \_____/
#                          /            |              |
#                     Gain bit   11 bits of data  Unused bits
#
#        Given the above representation, the data is not "packed" in the traditional
#        sense. For space saving, the DAQ WILL pack the data, removing the unused bits.
#
#        The data is then stored in the format:
#                         (NumAsics, NumPackedInts)
#        for each of the panels participating the DAQ.
#
#        Each detector panel has 6 asics, each with shape (168, 192). These are
#        arranged in the following format:
#
#                              A1   |   A3   |   A5
#                           --------+--------+--------
#                              A0   |   A2   |   A4
#
#        The raw retrieval function will deal with unpacking and reshaping. The final
#        output will be of shape:
#                         (NumPanels, 336, 576)
#        """
#        if evt is None:
#            return None
#
#        segs: Optional[dict[int, Any]] = self._segments(evt)
#
#        if segs is None:
#            return None
#        nsegs: int = len(segs)
#        n_asic_rows: int = 168
#        n_asic_cols: int = 192
#        # Final output will be as described above. (NPanels, 2*NRows, 3*NCols)
#        arr: npt.NDArray[np.uint16] = np.zeros(
#            (nsegs, n_asic_rows * 2, n_asic_cols * 3), dtype=np.uint16
#        )
#
#        # E.g. for 2 panels, we will have (2, 6, npixels)
#        # We will reshape it into (2, 192*2, 168*3)
#        # NOTE: This loop currently assumes that seg_idx starts at 0.
#        #       If not, then the enumerate segs_seen needs to be used for indexing
#        #       the output array. Or something needs to be decided in terms of how
#        #       to handle missing data in the event that DRP segment numbers don't start
#        #       at 0.
#        for _, seg_idx in enumerate(segs):
#            seg: psana.container.Container = segs[seg_idx]
#            unpacked: npt.NDArray[np.uint16] = self._unpackData(seg.raw)
#
#            # As a final step, reshape the data into a physical shape.
#            blocked_asics: npt.NDArray[np.uint16] = unpacked.reshape(
#                2, 3, n_asic_rows, n_asic_cols
#            )
#            arranged: npt.NDArray[np.uint16] = blocked_asics.transpose(
#                0, 2, 1, 3
#            ).reshape(n_asic_rows * 2, n_asic_cols * 3)
#
#            arr[seg_idx] = arranged
#
#        return arr
#
#
#    def _unpackData(
#        self, packed_data: npt.NDArray[np.uint16]
#    ) -> npt.NDArray[np.uint16]:
#        """Given a single panel's packed representation, unpack into 6*192*168 pixels.
#
#        Args:
#            packed_data (npt.NDArray[np.uint16]): The packed panel representation
#                as provided by the DAQ.
#
#        Returns:
#            unpacked_data (npt.NDArray[np.uint16]): An unpacked representation.
#                This function does not reshape the data and gives a flat representation
#                of the 6 asics.
#        """
#        raw_bytes_data: npt.NDArray[np.uint8] = packed_data.view(np.uint8)
#
#        num_pixels: int = (raw_bytes_data.size // 3) * 2
#
#        # Assert that we can decompose properly
#        assert raw_bytes_data.size % 3 == 0
#        assert num_pixels % 2 == 0
#
#        # Reshape into triplets of 3 bytes for our bitwise ops to unpack
#        triplets: npt.NDArray[np.uint8] = raw_bytes_data.reshape(-1, 3)
#        b0 = triplets[:, 0].astype(np.uint16)  # Cast now to avoid overflow
#        b1 = triplets[:, 1].astype(np.uint16)
#        b2 = triplets[:, 2].astype(np.uint16)
#
#        out: npt.NDArray[np.uint16] = np.empty(num_pixels, dtype=np.uint16)
#
#        out[0::2] = (b0 << 4) | (b1 >> 4)
#        out[1::2] = ((b1 & 0x0F) << 8) | b2
#
#        return out

    # EOF

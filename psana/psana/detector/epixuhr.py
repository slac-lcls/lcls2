#from time import time
#from psana.detector.NDArrUtils import info_ndarr
import numpy as np
from amitypes import Array2d, Array3d
import psana.detector.epix_base as eb
import logging
from psana.detector.detector_impl import DetectorImpl
logger = logging.getLogger(__name__)

is_none = eb.ut.is_none
M15 = 0o77777 # 15-bit mask
B16 = 0o100000 # the 16-th bit (counting from 1)

# make an empty detector interface for Matt's hardware
# configuration object so that config_dump works - cpo
class epixuhrhw_config_0_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhr_config_0_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhrhw_config_1_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhr_config_1_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhrhw_config_2_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhr_config_2_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

#from now on both hw and not have same version number
class epixuhrhw_config_2_1_1(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhr_config_2_1_1(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhrhw_config_3_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhr_config_3_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhrhw_config_3_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhr_config_3_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhr_config_3_2_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixuhrhw_config_3_2_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        
class epixuhr_raw_0_0_0(eb.epix_base):
    def __init__(self, *args, **kwargs):
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIXUHRASIC:V1')
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixuhr.data'
        self._segment_numbers = [0,1,2,3]
        self._nwarnings = 0
        self._nwarnings_max = kwargs.get('nwarnings_max', 5)

    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            #nx = segs[0].raw.shape[1]
            #ny = segs[0].raw.shape[0]
            f = segs[0].raw & self._data_bit_mask # 0x7fff
        return f

    def _cbits_config_segment(self, cob):
        """not used in epixuhr"""
        return None


    def _segment_ids(self):
        """Re-impliment epix_base._segment_ids for epixuhr
        returns list of detector segment ids using ASIC numbers, e.g.
        [00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-00,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-01,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-02,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-03]
         for det.raw._uniqueid: epixuhr_0016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206
         and self._segment_numbers = [0, 1, 2, 3]
        """
        id = self._uniqueid.split('_')[1] # 0016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206
        return ['%s-ASIC-%02d' % (id,i) for i in self._segment_numbers]


#####################
# PIETRO'S descrambler
# 11/07/24
#####################

    # Function to convert the 8bits values back into 12 bits elements
    def _pack_6_8bit_to_4_12bit_matrix(self,arr):
        # Ensure the input has 6 columns (i.e., shape is N x 6)
        if arr.shape[1] != 6:
            raise ValueError("Input array must have exactly 6 columns for each group.")

        # Step 1: Process each row of 6 bytes
        first_12bit = (arr[:, 0] << 4) | (arr[:, 1] >> 4)
        second_12bit = ((arr[:, 1] & 0x0F) << 8) | arr[:, 2]
        third_12bit = (arr[:, 3] << 4) | (arr[:, 4] >> 4)
        fourth_12bit = ((arr[:, 4] & 0x0F) << 8) | arr[:, 5]

        # Step 2: Stack results into a 2D array where each row has four 12-bit values
        result = np.column_stack((first_12bit, second_12bit, third_12bit, fourth_12bit))

        return result.astype(np.uint16)

    # Predefined lut to convert into the UHR lanemap
    def _lane_map_uhr35kHzv2(self, ):
        column_map = []
        cluster_map = np.empty(72)
        # Create the sp map
        sp_map = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # Create the cluster map
        for index in range(72):
            cluster_map[index] = index
        # Cast cluster_map to int
        cluster_map=cluster_map.astype(int)
        # Split the cluster in the 2 sp columns
        cluster_map = np.concatenate(
            [np.reshape(cluster_map[0:len(cluster_map) // 2], (12, 3)),
            np.flip(np.reshape(cluster_map[len(cluster_map) // 2:], (12, 3)),1)],axis=1)
        # Re-linearize the cluster_map
        cluster_map = np.reshape(cluster_map, 72)
        # Create the column map
        for cluster_n in range (14):
            column_map = np.append(column_map , (14 * cluster_map) + cluster_n)
        # Cast column_map to int
        column_map = column_map.astype(int)
        # Create the cluster_columns_map
        lane_map = np.concatenate(
            [np.reshape(column_map * 2, (168, 6)),
            np.reshape((column_map * 2) + 1, (168, 6))],axis=1)
        # Re-linearize the cluster_columns_map
        lane_map = np.reshape(lane_map, 2016)

        return lane_map

    # Function needed to descramble the raw frame
    def _descramble_raw_frame(self, raw_data, gain_msb):
        # Below is more or less a copy-paste of the `descramble` function from ePixViewer/asics/ePixUhr100kHz.py
        # Create the frames
        full_frame = np.empty((168, 192), dtype=int)
        full_frame_12bit = np.empty((1008, 32), dtype=int)
        num_pixels_x = 192
        num_pixels_y = 168
        num_data_bytes_per_frame = int(num_pixels_x*num_pixels_y*12/8)

        num_header_bytes_per_frame = 0
        num_total_bytes_per_frame = num_data_bytes_per_frame + num_header_bytes_per_frame

        # Remove bytes at the start. we should have 48384 bytes: 168*192*12 bit/8bit
        raw_data_adjusted = raw_data
        # Split into the 8 columns representing the 8 serial lanes
        raw_data_8bit = np.reshape(raw_data_adjusted, (6048, 8))
        for lanes in range(8):
            lane_12bit = np.empty((1008, 4), dtype='int')
            # Going back to the 64 bit
            lane_raw = raw_data_8bit[:, lanes]
            lane_48bit =np.flip(np.reshape(lane_raw,(1008,6)),1)
            lane_12bit = np.flip(self._pack_6_8bit_to_4_12bit_matrix(lane_48bit.astype(np.uint16)),1)
            full_frame_12bit[:, lanes * 4:(lanes * 4) + 4] = lane_12bit
        for columns in range(16):
            slice_lane = full_frame_12bit[:, columns * 2:(columns * 2) + 2]
            slice_lane = np.reshape(slice_lane, 2016)
            a = self._lane_map_uhr35kHzv2()

            column_tmp = slice_lane[a]
            column_tmp = np.flip(np.reshape(column_tmp, (168, 12)), 0)
            full_frame[:, columns * 12:(columns * 12) + 12] = column_tmp
        if gain_msb:
            full_frame = np.where(full_frame % 2 == 0, full_frame // 2, (full_frame - 1) // 2 + 2048)
        return full_frame

    def _descramble_3d_frames(self, all_raw_data, gain_msb=True):
        number_frames_rec = np.shape(all_raw_data)[0]
        data_3Darr = np.empty((number_frames_rec,168,192), dtype=np.uint16)
        # For loop over frames, can probably be done in numpy:
        for i, raw_data in enumerate(all_raw_data):
            pixels = self._descramble_raw_frame(raw_data=raw_data, gain_msb=gain_msb)
            data_3Darr[i] = pixels
        return data_3Darr

#####################

    def raw(self, evt) -> Array3d:
        if evt is None: return None
        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if segs is None: return None
        return self._descramble_3d_frames(segs[0].raw) # shape=(4, 192, 384)

    def calib(self, evt) -> Array3d: # already defined in epix_base and AreaDetectorRaw
        if self._nwarnings < self._nwarnings_max:
            self._nwarnings += 1
            s = 'TBD - calib IS NOT IMPLEMENTED YET!'
            if self._nwarnings == self._nwarnings_max: s += ' >>> STOP PRINT THESE WARNINGS'
            logger.warning(s)
        return self.raw(evt)

    def _calib_TBD(self, evt) -> Array3d: # already defined in epix_base and AreaDetectorRaw
        """ TBD - when pedestals are availavle..."""
        #logger.debug('%s.%s' % (self.__class__.__name__, sys._getframe().f_code.co_name))
        #print('TBD: %s.%s' % (self.__class__.__name__, sys._getframe().f_code.co_name))
        if evt is None: return None

        #t0_sec = time()
        raw = self.raw(evt)
        if is_none(raw, 'self.raw(evt) is None - return None'):
            return raw

        # Subtract pedestals
        peds = self._pedestals()
        if is_none(peds, 'det.raw._pedestals() is None - return det.raw.raw(evt)'):
            return raw
        #print(info_ndarr(peds,'XXX peds', first=1000, last=1005))

        gr1 = (raw & self._data_gain_bit) > 0

        #print(info_ndarr(gr1,'XXX gr1', first=1000, last=1005))
        pedgr = np.select((gr1,), (peds[1,:],), default=peds[0,:])
        arrf = np.array(raw & self._data_bit_mask, dtype=np.float32)
        arrf -= pedgr

        #print('XXX time for calib: %.6f sec' % (time()-t0_sec)) # 4ms on drp-neh-cmp001

        return arrf

epixuhr_raw_1_0_0 = epixuhr_raw_0_0_0
epixuhr_raw_2_0_0 = epixuhr_raw_1_0_0
epixuhr_raw_2_1_1 = epixuhr_raw_2_0_0
epixuhr_raw_3_0_0 = epixuhr_raw_2_1_1
epixuhr_raw_3_1_0 = epixuhr_raw_3_0_0
epixuhr_raw_3_2_0 = epixuhr_raw_3_1_0

#class epixuhr_raw_1_0_0(epixuhr_raw_0_0_0):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)

#class epixuhr_raw_2_0_0(epixuhr_raw_1_0_0):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)

#class epixuhr_raw_2_1_1(epixuhr_raw_2_0_0):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)

#class epixuhr_raw_3_0_0(epixuhr_raw_2_1_1):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)

#class epixuhr_raw_3_1_0(epixuhr_raw_3_0_0):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)

#    def image(self, evt, **kwargs) -> Array2d: # see in areadetector.py
#        if evt is None: return None
#        return self.raw(evt)[0].reshape(768,384)

# EOF

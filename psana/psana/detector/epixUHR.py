import os
import sys
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
class epixuhrhw_config_0_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super(epixuhrhw_config_0_0_0, self).__init__(*args)

class epixuhr_raw_0_0_0(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epixuhr_raw_0_0_0.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIXMASIC:V1')
        self._data_bit_mask = M15 # for epixm320 data on 2024-03-20 Dawood - epixM has 15 data bits.
        self._data_gain_bit = B16 # gain switching bit
        self._gain_bit_shift = 10
        self._gains_def = (-100.7, -21.3, -100.7) # ADU/Pulser
        self._gain_modes = ('SH', 'SL', 'AHL')
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixm320.data'
        self._dataDebug = None
        self._segment_numbers = [0,1,2,3]

    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            nx = segs[0].raw.shape[1]
            ny = segs[0].raw.shape[0]
            f = segs[0].raw & self._data_bit_mask # 0x7fff
        return f

    def _cbits_config_segment(self, cob):
        """not used in epixm"""
        return None


    def _segment_ids(self):
        """Re-impliment epix_base._segment_ids for epixm320
        returns list of detector segment ids using ASIC numbers, e.g.
        [00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-00,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-01,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-02,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-03]
         for det.raw._uniqueid: epixm320_0016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206
         and self._segment_numbers = [0, 1, 2, 3]
        """
        id = self._uniqueid.split('_')[1] # 0016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206
        return ['%s-ASIC-%02d' % (id,i) for i in self._segment_numbers]
    
    
    def read_uint12(self,data_chunk):
        fst_uint8, mid_uint8, lst_uint8 = np.reshape(data_chunk, (data_chunk.shape[0] // 3, 3)).astype(np.uint16).T
        fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
        snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
        return np.reshape(
            np.concatenate(
                (fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])


    def lane_map_uhr100(self,):
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
            np.flip(np.reshape(cluster_map[len(cluster_map) // 2:], (12, 3)),1)],
            axis=1)

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
            np.reshape((column_map * 2) + 1, (168, 6))],
            axis=1)

        # Re-linearize the cluster_columns_map
        lane_map = np.reshape(lane_map, 2016)

        return lane_map

    def get_pixels_from_raw_data(self,raw_data, gain_msb):  

        num_pixels_x = 192
        num_pixels_y = 168
        num_data_bytes_per_frame = int(num_pixels_x*num_pixels_y*12/8)

        num_header_bytes_per_frame = 0
        num_total_bytes_per_frame = num_data_bytes_per_frame + num_header_bytes_per_frame
  
        # Below is more or less a copy-paste of the `descramble` function from ePixViewer/asics/ePixUhr100kHz.py
        # Create the frames
        full_frame = np.empty((168, 192), dtype=int)
        full_frame_12bit = np.empty((1008, 32), dtype=int)
        print(f"RAW DATA SHAPE {np.shape(raw_data)}")
        # Remove bytes at the start. we should have 48384 bytes: 168*192*12 bit/8bit
        #raw_data_adjusted = raw_data[num_header_bytes_per_frame-16:48384 + num_header_bytes_per_frame-16]
        raw_data_adjusted = raw_data
        # Split into the 8 columns representing the 8 serial lanes
        raw_data_8bit = np.reshape(raw_data_adjusted, (6048, 8))
        
        for lanes in range(8):
            lane_48bit = np.empty((1008, 6))
            lane_12bit = np.empty((1008, 4), dtype='int')
            # Going back to the 64 bit
            lane_raw = raw_data_8bit[:, lanes]
            lane_64bit = np.flip(np.reshape(lane_raw, (756, 8)), 1)
            # Each row is now the output of the 48:64 gearbox.
            for i in range(252):
                lane_48bit[0 + i * 4, 0:6] = lane_64bit[0 + i * 3, 2:8]
                lane_48bit[1 + i * 4, 0:4] = lane_64bit[1 + i * 3, 4:8]
                lane_48bit[1 + i * 4, 4:6] = lane_64bit[0 + i * 3, 0:2]
                lane_48bit[2 + i * 4, 0:2] = lane_64bit[2 + i * 3, 6:8]
                lane_48bit[2 + i * 4, 2:8] = lane_64bit[1 + i * 3, 0:4]
                lane_48bit[3 + i * 4, 0:6] = lane_64bit[2 + i * 3, 0:6]

            # Now we need to shift from 8bit per entry to 12bit per entry
            for j in range (1008):
                lane_12bit[j, :] = self.read_uint12(lane_48bit[j, :])

            lane_12bit = np.flip(lane_12bit, 1)
            full_frame_12bit[:, lanes * 4:(lanes * 4) + 4] = lane_12bit
            
            for columns in range(16):
                slice_lane = full_frame_12bit[:, columns * 2:(columns * 2) + 2]
                slice_lane = np.reshape(slice_lane, 2016)
                a = self.lane_map_uhr100()

                column_tmp = slice_lane[a]
                column_tmp = np.flip(np.reshape(column_tmp, (168, 12)), 0)
                full_frame[:, columns * 12:(columns * 12) + 12] = column_tmp

            if gain_msb:
                full_frame = np.where(full_frame % 2 == 0, full_frame // 2, (full_frame - 1) // 2 + 2048)
        
        # Done
        return full_frame

    def get_data_3Darr(self,all_raw_data, gain_msb=True):
        
        number_frames_rec = np.shape(all_raw_data)[0]
        print(np.shape(all_raw_data)[0])
        print(np.shape(all_raw_data))
        data_3Darr = np.empty((number_frames_rec,168,192))
        
        # For loop over frames, can probably be done in numpy:
        for i, raw_data in enumerate(all_raw_data):
            pixels = self.get_pixels_from_raw_data(raw_data=raw_data, gain_msb=gain_msb)
            data_3Darr[i] = pixels
        return data_3Darr

    def raw(self, evt) -> Array3d: # see in areadetector.py
        if evt is None: return None
        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if segs is None: return None
        print(f"### shape of segs {np.shape(segs[0].raw)}")
        data_3Darr = self.get_data_3Darr(segs[0].raw)
        print("### 3D arr shape ###")
        print(np.shape(data_3Darr))
        return  data_3Darr # shape=(4, 192, 384)

    def calib(self, evt) -> Array3d: # already defined in epix_base and AreaDetectorRaw
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


#    def image(self, evt, **kwargs) -> Array2d: # see in areadetector.py
#        if evt is None: return None
#        return self.raw(evt)[0].reshape(768,384)

# EOF

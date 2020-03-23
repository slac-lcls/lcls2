#import bitstruct
import numpy as np
from collections import namedtuple
from psana.detector.detector_impl import DetectorImpl

def _to_word(data,shift=0,mask=0):
    rdata = data[::-1]
    w = 0
    for b in data[::-1]:
        w = w<<8
        w |= b
    w >>= shift
    if mask:
        w &= (1<<mask)-1
    return w

def _unpack(data):
    return ( _to_word(data[0:8]),       # pulseId
             _to_word(data[8:16]),      # timestamp
             _to_word(data[16:18],0,10),# fixed rate
             _to_word(data[16:18],10,0),# ac rate
             _to_word(data[18:20],0,3), # timeslot
             _to_word(data[18:20],3,0), # phase
             _to_word(data[20:22],0,1)==1, # beam_present
             0,                         # reserved1
             _to_word(data[20:22],4,4), # beam_destn
             0,                         # reserved2
             _to_word(data[22:24]),     # beam_charge
             _to_word(data[24:26]),     # beam_energy_0
             _to_word(data[26:28]),
             _to_word(data[28:30]),
             _to_word(data[30:32]),
             _to_word(data[32:34]),     # wavelen_0
             _to_word(data[34:36]),
             0,                         # reserved3
             _to_word(data[38:40]),     # mps_limit
             _to_word(data[40:48]))     # mps_power_class

class ts_ts_1_2_3(DetectorImpl):
    def __init__(self, *args):
        super(ts_ts_1_2_3, self).__init__(*args)

        fields = {
            'pulseId':'u64',
            'timestamp':'u64',
            'fixed_rate_markers':'u10',
            'ac_rate_markers':'u6',
            'ac_time_slot':'u3',
            'ac_time_slot_phase':'u13',
            'ebeam_present':'b1',
            'reserved1':'u3',
            'ebeam_destination':'u4',
            'reserved2':'u8',
            'requested_ebeam_charge_pc':'u16',
            'requested_ebeam_energy_loc1':'u16',
            'requested_ebeam_energy_loc2':'u16',
            'requested_ebeam_energy_loc3':'u16',
            'requested_ebeam_energy_loc4':'u16',
            'requested_photon_wavelength_sxu':'u16',
            'requested_photon_wavelength_hxu':'u16',
            'reserved3':'u16',
            'mps_limit':'u16', # one bit per destination
            'mps_power_class':'u64', # four bits per destination
        }

        self.TsData = namedtuple('tsdata',fields.keys())
        format_string = '>'
        total_len = 0
        for v in fields.values():
            format_string += v
            total_len += int(v[1:])
        format_string += '<' # indicated least-significant-byte first
        self.total_bytes=total_len//8

        #self.bitstructure = bitstruct.compile(format_string)

    def info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all TS data comes from one segment
        data = segments[0].data
        #unpacked = self.bitstructure.unpack(data.tobytes())
        unpacked = _unpack(data)
        return self.TsData(*unpacked)

    def sequencer_info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all TS data comes from one segment
        data = segments[0].data
        # look in the remaining bytes for the event codes
        tmp = data[self.total_bytes:]
        seq = tmp.view('uint16')
        return seq


class ts_ts_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(ts_ts_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

class ts_raw_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)

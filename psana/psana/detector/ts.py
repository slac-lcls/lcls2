import bitstruct
import numpy as np
from collections import namedtuple
from psana.detector.detector_impl import DetectorImpl

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
        format_string = ''
        total_len = 0
        for v in fields.values():
            format_string += v
            total_len += int(v[1:])
        format_string += '<' # indicated least-significant-byte first
        self.total_bytes=total_len//8

        self.bitstructure = bitstruct.compile(format_string)

    def info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all TS data comes from one segment
        data = segments[0].data
        unpacked = self.bitstructure.unpack(data.tobytes())
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
        # remove the two extra bytes at the end, perhaps caused by
        # pgp alignment issues?
        return seq[:-1]

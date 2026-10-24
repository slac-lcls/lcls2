#import bitstruct
"""Detector interfaces for the `ts` detector type (pulse id, timestamp, event codes and related fields) and for `triginfo`."""
import numpy as np
import typing
import amitypes
from collections import namedtuple
from psana.detector.detector_impl import DetectorImpl

def _to_word(data,shift=0,mask=0):
    rdata = data[::-1]
    w = 0
    for b in data[::-1]:
        w = w<<8
        w |= int(b)
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
    """Detector interface for detector type `ts`, software `ts`, version 1.2.3.

    `info` unpacks the first 48 bytes of segment 0's `data` into a `TsData` named tuple and
    `sequencer_info` returns the bytes after them viewed as uint16.
    """
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
        self._total_bytes=total_len//8

        #self.bitstructure = bitstruct.compile(format_string)

    def info(self,evt):
        # check for missing data
        """Return the data of segment 0 unpacked into a `TsData` named tuple, or None if the segments are missing.

        The fields (pulseId, timestamp, rate markers, time slot, ebeam and photon request values, MPS
        values) are decoded from little-endian words of `segments[0].data` by the module function
        `_unpack`; the reserved fields are always 0.
        """
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all TS data comes from one segment
        data = segments[0].data
        #unpacked = self.bitstructure.unpack(data.tobytes())
        unpacked = _unpack(data)
        return self.TsData(*unpacked)

    def sequencer_info(self,evt):
        # check for missing data
        """Return the bytes of segment 0's `data` after the first 48, viewed as uint16, or None if the segments are missing.

        The code comment says these bytes hold the event codes.
        """
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all TS data comes from one segment
        data = segments[0].data
        # look in the remaining bytes for the event codes
        tmp = data[self._total_bytes:]
        seq = tmp.view('uint16')
        return seq


class ts_ts_0_0_1(DetectorImpl):
    """Detector interface for detector type `ts`, software `ts`, version 0.0.1.

    Its private `_info` returns the first segment found in the event (not necessarily segment 0), or
    None if the segments are missing.
    """
    def __init__(self, *args):
        super(ts_ts_0_0_1, self).__init__(*args)
        #self._add_fields()

    def eventcodes(self,evt) -> amitypes.Array1d:
        """Return a list of 288 ints (0 or 1): bit i is bit `i & 0xf` of word `i >> 4` of `sequenceValues`.

        `sequenceValues` is read from the first segment. No missing-data check: raises
        AttributeError if the event does not have all of this detector's segments.
        """
        seqV = self._info(evt).sequenceValues
        return [int((seqV[i>>4]>>(i&0xf))&1) for i in range(288)]

    #def l1fid(self,evt) -> int:
    #    return self._info(evt).timeStamp&0x1ffff

    #def seconds(self,evt) -> float:
    #    ts = self._info(evt).timeStamp
    #    return float(ts>>32) + float(ts&0xffffffff)*1.e-9

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return next(iter(segments.values()))

class ts_raw_2_0_0(ts_ts_0_0_1):
    """Detector interface for detector type `ts`, software `raw`, version 2.0.0.

    Same as `ts_ts_0_0_1`; adds nothing.
    """
    def __init__(self, *args):
        super().__init__(*args)

class ts_raw_2_1_0(ts_raw_2_0_0):
    """Detector interface for detector type `ts`, software `raw`, version 2.1.0.

    Extends `ts_raw_2_0_0` with accessors for fields of the first segment.
    """
    def __init__(self, *args):
        super().__init__(*args)

    def inhibitCounts(self,evt) -> amitypes.Array1d:
        """Return the `inhibitCounts` field of the first segment.

        No missing-data check: raises AttributeError if the event does not have all of this detector's
        segments.
        """
        return self._info(evt).inhibitCounts

    def destination(self,evt) -> int:
        """Return the `ebeamDestn` field of the first segment.

        No missing-data check: raises AttributeError if the event does not have all of this detector's
        segments.
        """
        return self._info(evt).ebeamDestn

    def pulseId(self,evt) -> int:
        """Return the `pulseId` field of the first segment with bit 63 cleared.

        The code comment says bit 63 identifies LCLS-1 data. No missing-data check: raises AttributeError if the event does not have all of this detector's
        segments.
        """
        return self._info(evt).pulseId & ~(1<<63) # bit 63 identifies LCLS-1

    def timestamp(self,evt) -> int:
        """Return the `timeStamp` field of the first segment.

        No missing-data check: raises AttributeError if the event does not have all of this detector's
        segments.
        """
        return self._info(evt).timeStamp

class ts_cube_2_0_0(DetectorImpl):
    """Detector interface for detector type `ts`, software `cube`, version 2.0.0.

    On construction `_add_fields()` adds one method per data field declared in the config (except
    `software` and `version`); each returns that field of segment 0 for an event, or None if the
    detector's segments are missing.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._add_fields()
"""
    def bin(self, evt) -> amitypes.Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].bin
    def entries(self, evt) -> amitypes.Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].entries
"""

class triginfo_triginfo_0_0_1(DetectorImpl):
    """Detector interface for detector type `triginfo`, software `triginfo`, version 0.0.1.

    Each method returns bits of segment 0's `data`, or None if the segments are missing.
    """
    def __init__(self, *args):
        super(triginfo_triginfo_0_0_1, self).__init__(*args)

    def prescale(self, evt) -> int:
        """Return bit 0 of segment 0's `data`, or None if the detector's segments are missing."""
        segments = self._segments(evt)
        if segments is None: return None
        return (segments[0].data >> 0) & 0x1

    def persist(self, evt) -> int:
        """Return bit 1 of segment 0's `data`, or None if the detector's segments are missing."""
        segments = self._segments(evt)
        if segments is None: return None
        return (segments[0].data >> 1) & 0x1

    def monitor(self, evt) -> int:
        """Return bits 2 to 5 of segment 0's `data` (`(data >> 2) & 0xf`), or None if the detector's segments are missing."""
        segments = self._segments(evt)
        if segments is None: return None
        return (segments[0].data >> 2) & 0xf


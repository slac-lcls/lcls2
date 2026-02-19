import functools
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import numpy.typing as npt

from psana.detector.detector_impl import DetectorImpl
from amitypes import Array1d, Array2d

class wave8_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(wave8_raw_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

    def raw_all(self,evt) -> Array2d:
        # only return a 2D array if all channels present
        # have the same length
        waveforms = []
        for i in range(8):
            func = getattr(self,'raw_%d'%i, None)
            if func:
                wf = func(evt)
                if wf is None: continue # missing waveform
                if len(waveforms)>0:
                    if wf.shape != waveforms[-1].shape:
                        return None # waveforms not the same shape
                waveforms.append(wf)
        if len(waveforms)==0: return None # no waveforms present
        return np.stack(waveforms)

class wave8_fex_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(wave8_fex_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

    def base_all(self,evt) -> Array1d:
        vals = []
        for i in range(8):
            func = getattr(self,'base_%d'%i, None)
            if func:
                val = func(evt)
                if val is not None: vals.append(val)
        if len(vals)==0: return None # no data
        return np.array(vals)

    def integral_all(self,evt) -> Array1d:
        vals = []
        for i in range(8):
            func = getattr(self,'integral_%d'%i, None)
            if func:
                val = func(evt)
                if val is not None: vals.append(val)
        if len(vals)==0: return None # no data
        return np.array(vals)

class wave8_cube_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super(wave8_cube_2_0_0, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]
"""
    def bin(self, evt) -> Array1d:
        info = self._info(evt)
        if info is None: return None
        return info.bin

    def entries(self, evt) -> Array1d:
        info = self._info(evt)
        if info is None: return None
        return info.entries

    def raw_all(self,evt) -> Array3d:
        # only return a 3D array if all channels present
        # have the same length
        waveforms = []
        for i in range(8):
            func = getattr(self,'raw_%d'%i, None)
            if func:
                wf = func(evt)
                if wf is None: continue # missing waveform
                if len(waveforms)>0:
                    if wf.shape != waveforms[-1].shape:
                        return None # waveforms not the same shape
                waveforms.append(wf)
        if len(waveforms)==0: return None # no waveforms present
        return np.stack(waveforms)
"""

class wave8v1wf_config_1_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class wave8v1wf_raw_1_0_0(DetectorImpl):
    """Decoding of packed wave8v1 waveform from pvadetector."""

    class _SampleType(Enum):
        UINT16 = 0
        UINT32 = 1 # "32-bit" channels are actually 18-bit, but need wide-enough type

    def __init__(self, *args):
        super().__init__(*args)

        self._add_fields()

        self._sample_types: List[wave8v1wf_raw_1_0_0._SampleType] = []
        self._chan_enable: int = self._seg_configs()[0].config.read_only.ChanEnable.value
        self._delays: Dict[int, int] = {}
        self._nsamples: Dict[int, int] = {}
        self._offsets: Dict[int, int] = {}
        self._all_wf_same_length: bool = True

        last_nsamples: int = 0

        n_channels: int = 0
        # Seems there is some header at the beginning?
        # Think is an event header (14) + channel header (1)
        offset: int = 15
        for i in range(16):
            # Case here based on bit mask?
            if self._chan_enable & (1 << i):
                if i < 8:
                    self._sample_types.append(self._SampleType.UINT16)
                else:
                    self._sample_types.append(self._SampleType.UINT32)
                channel_delay_desc: Any = getattr(
                    self._seg_configs()[0].config.read_only, f"Delay{i}"
                )
                n_samples_desc: Any = getattr(
                    self._seg_configs()[0].config.read_only, f"NumberOfSamples{i}"
                )
                self._delays[i] = channel_delay_desc.value
                self._nsamples[i] = n_samples_desc.value
                if n_channels > 0:
                    #if self._nsamples[i] != self._nsamples[i-1]:
                    if self._nsamples[i] != last_nsamples:
                        self._all_wf_same_length = False
                self._offsets[i] = offset
                length: int = self._calc_length(i)
                offset += 1 + length # +1 for subsequent channel header

                n_channels += 1
                last_nsamples = self._nsamples[i]

                raw_func: functools.partial = functools.partial(self._raw_template, channel=i)
                setattr(self, f"raw_{i}", raw_func)

    def _calc_length(self, channel: int) -> int:
        """Returns the length of the data for a given channel in the RAW data."""
        length: int
        if self._sample_types[channel] == self._SampleType.UINT16:
            length = int(self._nsamples[channel]/2)
        else:
            length = self._nsamples[channel]
        return length

    # Need Array1d/2d type hints for AMI even though not actual type
    def _raw_template(self, evt, channel=0) -> Array1d:
        raw_arr: Optional[npt.NDArray[np.float64]] = self.value(evt)
        if raw_arr is None:
            return None

        # `:RAW` record field type is uint64, but channel access doesn't support
        # that so we get doubles (only 64 bit type)
        # Additionally, data fits in 32 bits so cast each array entry to uint32
        u32: npt.NDArray[np.uint32] = raw_arr.astype(np.uint32)
        if self._chan_enable & (1 << channel):
            offset: int = self._offsets[channel]
            length: int = self._calc_length(channel)
            # Reinterpret bytes at correct bit depth
            dtype: Type[Union[np.uint16, np.uint32]]
            if self._sample_types[channel] == self._SampleType.UINT32:
                # These channels are 18-bit
                dtype = np.uint32
            else:
                dtype = np.uint16
            return u32[offset:(offset+length)].view(dtype)
        return None

    def raw_all(self, evt) -> Array2d:
        if not self._all_wf_same_length:
            return None
        else:
            wfs: List[Union[npt.NDArray[np.uint16], npt.NDArray[np.uint32]]] = []
            for i in range(16):
                raw_func: functools.partial = getattr(self, f"raw_{i}")
                wf: Union[npt.NDArray[np.uint16], npt.NDArray[np.uint32], None]
                if (wf := raw_func(evt)) is not None:
                    wfs.append(wf)

            if len(wfs) == 0:
                return None
            return np.stack(wfs)

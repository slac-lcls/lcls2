"""Detector interfaces for wave8 data (raw waveforms, fex values, cube data) and the wave8v1 waveform decoder."""
import functools
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import numpy.typing as npt

from psana.detector.detector_impl import DetectorImpl
from amitypes import Array1d, Array2d

class wave8_raw_0_0_1(DetectorImpl):
    """Detector interface for detector type `wave8`, software `raw`, version 0.0.1.

    On construction `_add_fields()` adds one method per data field declared in the config (except
    `software` and `version`); each returns that field of segment 0 for an event, or None if the
    detector's segments are missing. `raw_all`
    stacks the waveforms of the generated `raw_0` to `raw_7` accessors.
    """
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
        """Return the waveforms of the generated `raw_0` to `raw_7` accessors stacked into a 2-D array.

        Accessors that do not exist and None waveforms are skipped. Returns None if no waveform is
        present or a waveform's shape differs from the previous one.
        """
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
    """Detector interface for detector type `wave8`, software `fex`, version 0.0.1.

    On construction `_add_fields()` adds one method per data field declared in the config (except
    `software` and `version`); each returns that field of segment 0 for an event, or None if the
    detector's segments are missing. `base_all`
    and `integral_all` collect the per-channel values.
    """
    def __init__(self, *args):
        super(wave8_fex_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

    def base_all(self,evt) -> Array1d:
        """Return a numpy array of the non-None values of the generated `base_0` to `base_7` accessors.

        Accessors that do not exist are skipped; returns None if there are no values.
        """
        vals = []
        for i in range(8):
            func = getattr(self,'base_%d'%i, None)
            if func:
                val = func(evt)
                if val is not None: vals.append(val)
        if len(vals)==0: return None # no data
        return np.array(vals)

    def integral_all(self,evt) -> Array1d:
        """Return a numpy array of the non-None values of the generated `integral_0` to `integral_7` accessors.

        Accessors that do not exist are skipped; returns None if there are no values.
        """
        vals = []
        for i in range(8):
            func = getattr(self,'integral_%d'%i, None)
            if func:
                val = func(evt)
                if val is not None: vals.append(val)
        if len(vals)==0: return None # no data
        return np.array(vals)

class wave8_cube_2_0_0(DetectorImpl):
    """Detector interface for detector type `wave8`, software `cube`, version 2.0.0.

    On construction `_add_fields()` adds one method per data field declared in the config (except
    `software` and `version`); each returns that field of segment 0 for an event, or None if the
    detector's segments are missing.
    """
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
    """Detector interface for detector type `wave8v1wf`, software `config`, version 1.0.0; adds nothing to `DetectorImpl`."""
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
        """Return the decoded waveforms of channels 0 to 15 stacked into a 2-D array, or None.

        Returns None if the configured channels have different sample counts or no waveform is decoded.
        It calls `raw_<i>` for every i from 0 to 15 without a default, and the constructor creates these
        accessors only for enabled channels, so a disabled channel raises AttributeError (unless
        `_add_fields` created an attribute of that name).
        """
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

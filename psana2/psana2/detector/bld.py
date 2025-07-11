import numpy as np
import numpy.typing as npt
from psana2.detector.detector_impl import DetectorImpl
# for AMI
import typing
import inspect

class hpsex_hpsex_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(hpsex_hpsex_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

class hpscp_hpscp_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(hpscp_hpscp_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        return segments[0]

class ebeam_ebeamAlg_0_7_1(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_ebeamAlg_0_7_1, self).__init__(*args)
        self._add_fields()

class ebeam_raw_2_0_0(ebeam_ebeamAlg_0_7_1):
    def __init__(self, *args):
        super().__init__(*args)

class ebeamh_raw_2_0_0(ebeam_ebeamAlg_0_7_1):
    def __init__(self, *args):
        super().__init__(*args)

class gasdet_raw_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super(gasdet_raw_1_0_0, self).__init__(*args)
        self._add_fields()

class bmmon_raw_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super(bmmon_raw_1_0_0, self).__init__(*args)
        self._add_fields()

class pcav_raw_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super(pcav_raw_2_0_0, self).__init__(*args)
        self._add_fields()

class pcavh_raw_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super(pcavh_raw_2_0_0, self).__init__(*args)
        self._add_fields()

class gmd_raw_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super(gmd_raw_2_0_0, self).__init__(*args)
        self._add_fields()

class gmd_raw_2_1_0(DetectorImpl):
    def __init__(self, *args):
        super(gmd_raw_2_1_0, self).__init__(*args)
        self._add_fields()

class xgmd_raw_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super(xgmd_raw_2_0_0, self).__init__(*args)
        self._add_fields()

class xgmd_raw_2_1_0(DetectorImpl):
    def __init__(self, *args):
        super(xgmd_raw_2_1_0, self).__init__(*args)
        self._add_fields()

class feespec_raw_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
        self._add_fields()

class usdusb_raw_1_0_0(DetectorImpl):
    def __init__(self, *args):
        """Fast USB encoder.

        There are 4 channels.

        Hidden fields:
            Config
                _countMode: uint32_t[4]
                _quadMode:  uint32_t[4]
            FEX Config
                _offset: int32_t[4]
                _scale:  double[4]
                _name:   int8_t[4][48] (actually str[4])
            Raw Data
                _header:    uint8_t[6]
                _din:       uint8_t
                _estop:     uint8_t
                _timestamp: uint32_t
                _count:     uint32_t[4]
                _status:    uint8_t[4]
                _ain:       uint16_t[4]
        Visible field:
            FEX:
                encoder_values: double[4] - (raw_count + offset)*scale
        """
        super().__init__(*args)
        self._add_fields()

    def descriptions(self, evt) -> typing.List[str]:
        """Unpack chars into a list of descriptions per channel"""
        # Holds [4][48] array of chars with descriptive name
        # 4 channels, with max 48 chars per channel description
        str_bytes: npt.NDArray[np.int8] = self._name(evt)
        ch_descs: typing.List[str] = []
        for i in range(4):
            ch_desc: str = str_bytes[i*48:(i+1)*48].tobytes().decode().strip("\x00")
            ch_descs.append(ch_desc)
        return ch_descs

class scbld_raw_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super(scbld_raw_1_0_0, self).__init__(*args)
        self._add_fields_and_severity()

    def _add_fields_and_severity(self):
        for config in self._configs:
            if not hasattr(config,'software'): continue

            seg      = getattr(config.software,self._det_name)
            seg_dict = getattr(seg[0],self._drp_class_name)
            attrs    = [attr for attr in vars(seg_dict) if not (attr=='software' or attr=='version' or attr=='severity')]
            for i,field in enumerate(attrs):
                fd = getattr(seg_dict,field)
                # fd._type, fd._rank
                def func(evt, field=field) -> self._return_types(fd._type,fd._rank):
                    info = self._info(evt)
                    if info is None:
                        return None
                    else:
                        return getattr(info,field)
                setattr(self, field, func)

                def func(evt, field=field, i=i) -> int:
                    info = self._info(evt)
                    if info is None:
                        return None
                    else:
                        return (getattr(info,'severity')>>(i*2))&3
                setattr(self, f'{field}_sevr', func)


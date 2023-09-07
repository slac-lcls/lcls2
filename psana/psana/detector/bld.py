import numpy as np
from psana.detector.detector_impl import DetectorImpl
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


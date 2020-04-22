import typing
import amitypes
from psana.dgram import Dgram

class Container(object):
    def __init__(self):
        pass

class DetectorImpl(object):
    def __init__(self, det_name, drp_class_name, configinfo, calibconst):
        self._det_name       = det_name
        self._drp_class_name = drp_class_name

        self._configs = configinfo.configs # configs for only this detector
        self._sorted_segment_ids = configinfo.sorted_segment_ids
        self._uniqueid = configinfo.uniqueid
        self._dettype = configinfo.dettype
        
        self._calibconst     = calibconst # only my calibconst (equivalent to det.calibconst['det_name'])

    def _segments(self,evt):
        """
        Look in the event to find all the dgrams for our detector/drp_class
        e.g. (xppcspad,raw) or (xppcspad,fex)
        """
        key = (self._det_name,self._drp_class_name)
        if key in evt._det_segments:
            # check that all promised segments have been received
            evt_segments = list(evt._det_segments[key].keys())
            evt_segments.sort()
            if evt_segments != self._sorted_segment_ids:
                return None
            else:
                return evt._det_segments[key]
        else:
            return None

    def _info(self,evt):
        return self._segments(evt)[0]

    @staticmethod
    def _return_types(rtype,rrank):
        rval = typing.Any
        if rtype<8:
            if rrank==0:
                rval = int
            elif rrank==1:
                rval = amitypes.Array1d
            elif rrank==2:
                rval = amitypes.Array2d
            elif rrank==3:
                rval = amitypes.Array3d
        elif rtype<10:
            rval = float
        elif rtype<11:
            rval = str
        return rval

    def _add_fields(self):
        for config in self._configs:
            if not hasattr(config,'software'): continue
            
            seg      = getattr(config.software,self._det_name)
            seg_dict = getattr(seg[0],self._drp_class_name)
            attrs    = [attr for attr in vars(seg_dict) if not (attr=='software' or attr=='version')]
            for field in attrs:
                fd = getattr(seg_dict,field)
                # fd._type, fd._rank
                def func(evt, field=field) -> self._return_types(fd._type,fd._rank):
                    info = self._info(evt)
                    if info is None:
                        return None
                    else:
                        return getattr(info,field)
                setattr(self, field, func)


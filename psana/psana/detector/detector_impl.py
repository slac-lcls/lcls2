import typing
import amitypes

class DetectorImpl(object):
    def __init__(self, det_name, drp_class_name, configs, calibconst):
        self._det_name       = det_name
        self._drp_class_name = drp_class_name
        self._configs        = configs
        self._calibconst     = calibconst

        self._config_segments = []
        for config in self._configs:
            # mona put this in since epics right now only exists
            # in one xtc file.
            if hasattr(config.software,self._det_name):
                seg_dict = getattr(config.software,self._det_name)
                self._config_segments += list(seg_dict.keys())
        self._config_segments.sort()
        return

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
            if evt_segments != self._config_segments:
                return None
            else:
                return evt._det_segments[key]
        else:
            return None

    def _return_types(self,rtype,rrank):
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
            if hasattr(config.software,self._det_name):
                seg      = getattr(config.software,self._det_name)
                seg_dict = getattr(seg[0],self._drp_class_name)
                attrs    = [attr for attr in vars(seg_dict) if not (attr=='software' or attr=='version')]
                for field in attrs:
                    fd = getattr(seg_dict,field)
                    # fd._type, fd._rank
                    def func(evt, field=field) -> self._return_types(fd._type,fd._rank):
                        return getattr(self._info(evt),field)
                    setattr(self, field, func)

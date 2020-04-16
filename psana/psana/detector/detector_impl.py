import typing
import amitypes
from psana.dgram import Dgram

class Container(object):
    def __init__(self):
        pass

class DetectorImpl(object):
    def __init__(self, det_name, drp_class_name, all_det_configs, calibconst):
        self._det_name       = det_name
        self._drp_class_name = drp_class_name

        # Setup a new config list - for configs missing this detector,
        # add blank dictionary as a placeholder.
        self._configs = [Dgram(view=config) for config in all_det_configs]
        self._config_segments = []
        # a dictionary of the ids (a.k.a. serial-number) of each segment
        self._segids = {}
        for config in self._configs:
            if hasattr(config.software, self._det_name): 
                seg_dict = getattr(config.software, self._det_name)
                self._config_segments += list(seg_dict.keys())
                #e.g. self._configs[0].software.xpphsd[0].detid
                seg_dict = getattr(config.software, self._det_name)
                for segment, det in seg_dict.items():
                    self._segids[segment]=det.detid
                    self._dettype=det.dettype
            else:
                config.__dict__ = {self._det_name: {}}
        self._config_segments.sort()

        self._get_uniqueid()

        self._calibconst     = calibconst # only my calibconst (equivalent to det.calibconst['det_name'])

    def _get_uniqueid(self):
        segids = self._segids
        self._uniqueid = self._dettype
        for sorted_segid in sorted(segids):
            self._uniqueid+='_'+segids[sorted_segid]

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
            if hasattr(config.software,self._det_name):
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


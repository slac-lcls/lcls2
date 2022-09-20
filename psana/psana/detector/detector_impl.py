import typing
import amitypes
from psana.dgram import Dgram


def hiddenmethod(obj):
    """
    Adds an '_hidden' attribute to an object so it won't be picked up by detinfo
    """
    obj._hidden = True
    return obj



class Container(object):
    def __init__(self):
        pass


class MissingDet:
    def __init__(self):
        pass
    def __getattr__(self, name):
        """ Returns itself recursively """
        return MissingDet()
    def __iter__(self):
        return self
    def __next__(self):
        """ Returns an empty iterator """
        raise StopIteration
    def __call__(self, evt): # support only one arg - following detector interface proposal
        return None


class DetectorImpl(object):
    def __init__(self, det_name, drp_class_name, configinfo, calibconst,
            env_store   = None,
            var_name    = None):
        self._det_name          = det_name
        self._drp_class_name    = drp_class_name

        # Both configs and calibconst are for only this detector
        self._configs           = configinfo.configs
        self._calibconst        = calibconst
        self._sorted_segment_ids= configinfo.sorted_segment_ids
        self._uniqueid          = configinfo.uniqueid
        self._dettype           = configinfo.dettype
        self._env_store         = env_store
        self._var_name          = var_name

    def _seg_configs(self):
        """
        Gather up all the segment configs into an easier-to-use dictionary
        """
        seg_configs = {}
        for dgram in self._configs:
            if hasattr(dgram,self._det_name):
                seg_configs.update(getattr(dgram,self._det_name))
        return seg_configs

    def config(self, evt):
        return self._seg_configs()

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
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        return segments[0]

    @staticmethod
    def _return_types(rtype,rrank):
        rval = typing.Any
        if rtype<10:
            if rrank==0:
                if rtype<8:
                    rval = int
                else:
                    rval = float
            elif rrank==1:
                rval = amitypes.Array1d
            elif rrank==2:
                rval = amitypes.Array2d
            elif rrank==3:
                rval = amitypes.Array3d
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


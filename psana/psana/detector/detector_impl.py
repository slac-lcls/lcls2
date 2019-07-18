class DetectorImpl(object):
    def __init__(self, det_name, drp_class_name, configs, calibs):
        self._det_name       = det_name
        self._drp_class_name = drp_class_name
        self._configs        = configs
        self._calibs         = calibs

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

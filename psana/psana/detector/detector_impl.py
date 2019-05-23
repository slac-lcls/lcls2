class DetectorImpl(object):
    def __init__(self, det_name, drp_class_name, configs, calibs):
        self._det_name       = det_name
        self._drp_class_name = drp_class_name
        self._configs        = configs
        self._calibs         = calibs
        return
    def _activeChannels(self,config):
        """
        returns a list of active channels for a particular segment, using only the
        config information. should be overridden by detector-specific code where
        appropriate.  this should perhaps be changed to be pure virtual.
        """
        return [0]
    def _channels(self):
        """
        returns a dictionary with segment numbers as the key, and a list
        of active channel numbers for each segment.
        """
        channels = {}
        for config in self._configs:
            seg_dict = getattr(config,self._det_name)
            for key in seg_dict:
                channels[key] = self._activeChannels(config)
        return channels
    def _segments(self,evt):
        """
        Look in the event to find all the dgrams for our detector/drp_class
        e.g. (xppcspad,raw) or (xppcspad,fex)
        """
        key = (self._det_name,self._drp_class_name)
        if key in evt._det_segments:
            return evt._det_segments[key]
        else:
            return None

class DetectorImpl(object):
    def __init__(self, det_name, drp_class_name, configs, calibs):
        self._det_name       = det_name
        self._drp_class_name = drp_class_name
        self._configs        = configs
        self._calibs         = calibs
        return
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

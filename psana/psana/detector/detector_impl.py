class DetectorImpl(object):
    def __init__(self, det_name, drp_class_name, configs, calibs):
        self._name           = det_name
        self._drp_class_name = drp_class_name
        self._configs        = configs
        self._calibs         = calibs
        return
    def dgrams(self,evt):
        """
        Look in the event to find all the dgrams for our detect/drp_class
        e.g. (xppcspad,raw) or (xppcspad,fex)
        """
        return evt._det_dgrams[(self._name,self._drp_class_name)]

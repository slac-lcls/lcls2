from psana.detector.detector_impl import DetectorImpl

# create a dictionary that can be used to look up other
# information about an epics variable.  the key in
# the dictionary is the "detname" of the epics variable
# (from "detnames -e") which can be either the true (ugly)
# epics name or a nicer user-defined name. the most obvious
# way to use this information would be to retrieve the
# real epics name from the "nice" user-defined detname.

class epicsinfo_epicsinfo_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
        self._infodict={}
        for c in self._configs:
            if hasattr(c,'epicsinfo'):
                for seg,value in c.epicsinfo.items():
                    names = getattr(value,'epicsinfo')
                    keys = names.keys.split(',')
                    for n in dir(names):
                        if n.startswith('_') or n=='keys': continue
                        if n not in self._infodict: self._infodict[n]={}
                        values = getattr(names,n).split('\n')
                        for k,v in zip(keys,values): self._infodict[n][k]=v
    def __call__(self):
        return self._infodict

class pv_raw_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
        self._add_fields()

class encoder_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def value(self,evt):
        """
        From Zach:
        “Scale” is copied from the configured “units per encoderValue”
        configuration. So a scale of “6667” means we have 6667 units per count.
        What is the unit? Well, I coded it to be “nanometers per count”
        for the average axis that is controlled in millimeters, since for
        most axes this ends up being an integer, but this axis is controlled
        in microradians, so that “normal case” conversion doesn’t really fit.
        Checking the code, the configured scale for the encoder is 0.0066667
        urads/count. I guess that makes sense since you’d usually multiply
        by 1e-6 to go from mm to nm.
 
        So to get the “real motor position” you would in general do:
        Position (real units) = encoderValue * scale * 1e-6
        Which works for the current encoderValue and the real position
        shown in the screen.
 
        Though clearly the scale is rounded in this case, with a true
        value of 2/3 * 1e4

        The Controls group has conventions of mm/urad for linear/rotary
        motion units, so this routine returns urad for the rix mono encoder.
        """
        segments = self._segments(evt)
        if segments is None: return None
        return segments[0].encoderValue*segments[0].scale*1e-6

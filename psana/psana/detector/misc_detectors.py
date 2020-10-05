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
        self.infodict={}
        for c in self._configs:
            if hasattr(c,'epicsinfo'):
                for seg,value in c.epicsinfo.items():
                    names = getattr(value,'epicsinfo')
                    keys = names.keys.split(',')
                    for n in dir(names):
                        if n.startswith('_') or n=='keys': continue
                        if n not in self.infodict: self.infodict[n]={}
                        values = getattr(names,n).split('\n')
                        for k,v in zip(keys,values): self.infodict[n][k]=v
    def __call__(self):
        return self.infodict

class pv_raw_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
        self._add_fields()

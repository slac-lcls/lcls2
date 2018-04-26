# source /reg/g/psdm/etc/psconda.sh
from psana import *
import json
import cPickle as pickle
import numpy as np
import inspect, base64

class parse_dgram():
    def __init__(self, ds, source, detector, config, event_limit):
        self.events=[]
        self.event_limit = event_limit
        self.ds = ds
        self.source = source
        self.detector = detector
        self.config = config
        self.ignored_keys = ['common_mode','bit_length', 'conjugate', 'denominator', 'imag', 'numerator', 'real']
        self.end_types = (int, float, np.ndarray, list)

        self.process_dgram()

    def strip_list(self, input_list, prepend=''):
        filt_list = list(filter(lambda x: x[:2] != '__' and\
                                x not in self.ignored_keys, input_list))
        if prepend:
            filt_list = list(map(lambda x: prepend+'__' +x, filt_list))
        return filt_list
    # We need a special version of get attr
    # The fancy functional version works for regular classes (aa.bb.cc..)
    # Accessing with methods requires the messier looped version (aa().bb.cc()...)
    def getattr(self,obj, attr):
        try:
            value = reduce(getattr, attr.split('__'), obj)
        except AttributeError:
            value = obj
            attrs = attr.split('__')
            for att in attrs[:-1]:
                it_meth = att.split('##')
                if len(it_meth)>1:
                    it_v = int(it_meth[1])
                    value = getattr(value,it_meth[0])(it_v)
                else:
                    value = getattr(value,att)()
            value = getattr(value, attrs[-1])
        return value

    def parse_event(self, dgram):
        config_keys = dir(dgram)
        config_keys = self.strip_list(config_keys)
        config_dict = {}

        for key in config_keys:
            new_keys = []
            tstat = self.parse_key(dgram, key)
            if tstat[0] == "End value":
                # This is the end point of the parsing.
                # Convert any np arrays to base64 for json
                outkey = key.replace('__', '_').replace('##','')
                config_dict[outkey] = self.bitwise_array(tstat[1])

            elif tstat[0] == "Method":
                new_keys = dir(self.getattr(dgram,key)())
                new_keys = self.strip_list(new_keys, key)
                if new_keys:
                    # print("Keys found in %s method are: " % key, new_keys)
                    config_keys += new_keys
            elif tstat[0] == "Indexed method":

                # Kluge for CsPad quad() segfault instead of IndexError
                if self.detector == CsPad.DataV2:
                    it_limit = 4
                else:
                    it_limit = 128
                try:
                    for i in range(it_limit):
                        new_keys = dir(self.getattr(dgram,key)(i))
                        new_keys = self.strip_list(new_keys, key+"##%i" % i)
                        if new_keys:
                            # print("Keys found in %s%i method are: " % (key,i), new_keys)
                            config_keys += new_keys
                except IndexError:
                    pass
            elif tstat[0] == "List of methods":
                # new_keys = self.getattr(dgram,key)
                # new_keys = self.strip_list(new_keys, key)
                pass
        return config_dict

    def parse_key(self,dgram, key):

            try:
                cdk = self.getattr(dgram, key)()
                iterable_method = False
            except Exception as e:
                if str(e).startswith('Python argument types in'):
                    iterable_method = True
                    return "Indexed method", False
                else:
                    cdk = self.getattr(dgram, key)

            if type(cdk) is list and type(cdk[0]) not in self.end_types:
                return "List of methods", False

            # Try to cast enum types to ints
            try:
                cdk = int(cdk)
            except:
                pass

            if type(cdk) in self.end_types:
                return "End value", cdk
            else:
                return "Method", False

    def bitwise_array(self, value):
        if np.isscalar(value):
            return value
        val = np.asarray(value)
        return [base64.b64encode(val), val.shape, val.dtype.str]


    def process_dgram(self):
        cs = self.ds.env().configStore()
        configure = cs.get(self.config, self.source)
        self.events.append(self.parse_event(configure))
        for ct,evt in enumerate(self.ds.events()):
            if ct>self.event_limit:
                break
            framev2 = evt.get(self.detector, self.source)
            evtd = self.parse_event(framev2)

            self.events.append(evtd)

# Examples
event_limit = 10

## Jungfrau
ds = DataSource('exp=xpptut15:run=430')
source = Source('DetInfo(MfxEndstation.0:Jungfrau.0)')
detector = Jungfrau.ElementV2
config = Jungfrau.ConfigV3

cfgd = parse_dgram(ds, source, detector, config, event_limit)
with open("jungfrau.json", 'w') as f:
    f.write(json.dumps(cfgd.events))

# cs = ds.env().configStore()
# conf = cs.get(config, source)


## Epix
ds = DataSource('exp=xpptut15:run=260')
source = Source('DetInfo(XcsEndstation.0:Epix100a.1)')
detector = Epix.ElementV3
config = Epix.Config100aV2

cfgd = parse_dgram(ds, source, detector, config, event_limit)
with open("epix.json", 'w') as f:
    f.write(json.dumps(cfgd.events))


## Crystal dark runs

ds = DataSource('exp=cxid9114:run=89')
source = Source('DetInfo(CxiDs1.0:Cspad.0)')
detector = CsPad.DataV2
config = CsPad.ConfigV5

cfgd = parse_dgram(ds, source, detector, config, event_limit)
with open("crystal_dark.json", 'w') as f:
    f.write(json.dumps(cfgd.events))

## Crystal x-ray runs

ds = DataSource('exp=cxid9114:run=95')
source = Source('DetInfo(CxiDs1.0:Cspad.0)')
detector = CsPad.DataV2
config = CsPad.ConfigV5

cfgd = parse_dgram(ds, source, detector, config, event_limit)
with open("crystal_xray.json", 'w') as f:
    f.write(json.dumps(cfgd.events))

cs = ds.env().configStore()
conf = cs.get(config, source)

evt = ds.events().next()
evt = ds.events().next()

framev2 = evt.get(detector, source)

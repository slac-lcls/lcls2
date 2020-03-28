#!/usr/bin/env python

# Remember to run on psana machine and source /reg/g/psdm/etc/psconda.sh
# Then run translate_xtc_demo.py on psbuild-rhel7 and source setup_env.sh
#
# In LCLS1 environment run it:
# python conv-01-xtc1-to-json.py [number]

from psana import *
import json
import cPickle as pickle
import numpy as np
import inspect, base64

#--------------------

class parse_dgram():
    def __init__(self, ds, source, detector, config, event_limit):
        self.events=[]
        self.event_limit = event_limit
        self.ds = ds
        self.source = source
        self.detector = detector
        self.config = config
        self.ignored_keys = ['common_mode', 'bit_length', 'conjugate', 'denominator', 'imag', 'numerator', 'real']
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
                if not tstat[1] : continue # skip it if False
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

    def parse_key(self, dgram, key):
        try:
            cdk = self.getattr(dgram, key)()
            iterable_method = False
        except Exception as e:
            if str(e).startswith('Python argument types in'):
                iterable_method = True
                return "Indexed method", False
            else:
                cdk = self.getattr(dgram, key)

        #print 'type(cdk): ', type(cdk)
        #print 'cdk: ', cdk
        if type(cdk) is list and len(cdk)==0:
            return "List of methods is empty", False

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

#--------------------
# Examples
#--------------------

def parse_xtc(exp, run, detname, dettype, datatype, configtype, event_limit=10) :
    nev = event_limit
    ofname = 'data-%s-r%04d-e%06d-%s.json' % (exp,run,nev,dettype)
    dsname = 'exp=%s:run=%d'%(exp,run)
    print 'dsname %s' % dsname
    print 'Save data in file %s' % ofname

    ds = DataSource(dsname)
    source = Source(detname)

    cfgd = parse_dgram(ds, source, datatype, configtype, event_limit)
    with open(ofname, 'w') as f:
        f.write(json.dumps(cfgd.events))

#--------------------

def parse_xtc_file(xtcfname, exp, run, detname, dettype, datatype, configtype, event_limit=10) :
    nev = event_limit
    ofname = 'data-%s-r%04d-e%06d-%s.json' % (exp,run,nev,dettype)
    dsname = 'exp=%s:run=%d'%(exp,run)
    print 'xtcfname %s' % xtcfname
    print 'dsname %s' % dsname
    print 'Save data in file %s' % ofname

    #ds = DataSource(dsname)
    ds = DataSource(xtcfname)
    source = Source(detname)

    cfgd = parse_dgram(ds, source, datatype, configtype, event_limit)
    with open(ofname, 'w') as f:
        f.write(json.dumps(cfgd.events))

#--------------------

def parse_xpptut15_jungfrau() :
    # event_keys -d exp=xpptut15:run=430 -m3
    parse_xtc('xpptut15', 430, 'MfxEndstation.0:Jungfrau.0', 'jungfrau',\
              Jungfrau.ElementV2, Jungfrau.ConfigV3, event_limit=10)

#--------------------

def parse_cxid9114_cspad() :
    # event_keys -d exp=cxid9114:run=89 -m3
    parse_xtc('cxid9114', 89, 'CxiDs1.0:Cspad.0', 'cspad',\
              CsPad.DataV2, CsPad.ConfigV5, event_limit=10)

#--------------------

def parse_cxid9114_cspad2() :
    # event_keys -d exp=cxid9114:run=89 -m3
    parse_xtc('cxid9114', 89, 'CxiDs2.0:Cspad.0', 'cspad',\
              CsPad.DataV2, CsPad.ConfigV5, event_limit=10)

#--------------------

def parse_xpptut15_pnccd() :
    # event_keys -d exp=xpptut15:run=450 -m3
    # /reg/d/psdm/xpp/xpptut15/xtc/
    parse_xtc('xpptut15', 450, 'Camp.0:pnCCD.1', 'pnccd',\
              PNCCD.FramesV1, PNCCD.ConfigV2, event_limit=10)

    # event_keys -d exp=sxrx20915:run=206 -m3
    # /reg/d/psdm/sxr/sxrx20915/xtc/
    #parse_xtc('sxrx20915', 206, 'Camp.0:pnCCD.1', 'pnccd',\
    #          PNCCD.FramesV1, PNCCD.ConfigV2, event_limit=10)

#--------------------

def parse_xppl1316_opal1k() :
    # /reg/g/psdm/detector/data_test/types/ -> /reg/g/psdm/detector/data_test/xtc/
    # 0013-SxrBeamline.0-Opal1000.1.xtc   -> sxrb6813-e363-r0034-s00-c00.xtc
    # 0015-Opal8000_FrameV1.xtc           -> test_014_mec_meca6013_e355-r0009-s00-c00.xtc
    # 0018-MfxEndstation.0-Opal4000.0.xtc -> mfxm5116-e789-r0020-s00-c00.xtc
    # 0019-XppEndstation.0-Opal1000.1.xtc -> xppl1316-e750-r0193-s00-c00.xtc
    # 0020-XppEndstation.0-Opal1000.1.xtc -> xppn4116-e851-r0137-s00-c00.xtc
    # 0022-AmoEndstation.0-Opal1000.1.xtc -> amo11816-e934-r0025-s00-c00.xtc
    # 0022-AmoEndstation.0-Opal1000.2.xtc -> amo11816-e934-r0025-s00-c00.xtc

    # event_keys -d /reg/g/psdm/detector/data_test/xtc/xppl1316-e750-r0193-s00-c00.xtc -m3
    # event_keys -d /reg/g/psdm/detector/data_test/types/0019-XppEndstation.0-Opal1000.1.xtc -m3

    fname = '/reg/g/psdm/detector/data_test/xtc/xppl1316-e750-r0193-s00-c00.xtc'
    parse_xtc_file(fname, 'xppl1316', 193, 'XppEndstation.0:Opal1000.1', 'opal1k',\
                   Camera.FrameV1, Opal1k.ConfigV1, event_limit=10)

#--------------------

def parse_xppn4116_opal1k() :
    # event_keys -d /reg/g/psdm/detector/data_test/xtc/xppn4116-e851-r0137-s00-c00.xtc -m3

    fname = '/reg/g/psdm/detector/data_test/xtc/xppn4116-e851-r0137-s00-c00.xtc'
    parse_xtc_file(fname, 'xppn4116', 137, 'XppEndstation.0:Opal1000.1', 'opal1k',\
                   Camera.FrameV1, Opal1k.ConfigV1, event_limit=10)

#--------------------

def parse_mfxm5116_opal4k() :
    #    /reg/g/psdm/detector/data_test/types/0018-MfxEndstation.0-Opal4000.0.xtc
    # -> /reg/g/psdm/detector/data_test/xtc/mfxm5116-e789-r0020-s00-c00.xtc
    # event_keys -d /reg/g/psdm/detector/data_test/xtc/mfxm5116-e789-r0020-s00-c00.xtc -m3
    fname = '/reg/g/psdm/detector/data_test/xtc/mfxm5116-e789-r0020-s00-c00.xtc'
    parse_xtc_file(fname, 'mfxm5116', 20, 'MfxEndstation.0:Opal4000.0', 'opal4k',\
                   Camera.FrameV1, Opal1k.ConfigV1, event_limit=10)
                   #Camera.FrameV1, Camera.FrameFexConfigV1, event_limit=10)

#--------------------

def parse_meca6013_opal8k() :
    #    /reg/g/psdm/detector/data_test/types/0015-Opal8000_FrameV1.xtc 
    # -> /reg/g/psdm/data_test/Translator/test_014_mec_meca6013_e355-r0009-s00-c00.xtc
    # event_keys -d /reg/g/psdm/data_test/Translator/test_014_mec_meca6013_e355-r0009-s00-c00.xtc -m3
    fname = '/reg/g/psdm/data_test/Translator/test_014_mec_meca6013_e355-r0009-s00-c00.xtc'
    parse_xtc_file(fname, 'meca6013', 9, 'MecTargetChamber.0:Opal8000.1', 'opal8k',\
                   Camera.FrameV1, Opal1k.ConfigV1, event_limit=10)

#--------------------

if __name__ == "__main__" :
    import sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'

    if   tname == '0': parse_xpptut15_jungfrau()
    elif tname == '1': parse_cxid9114_cspad()
    elif tname == '2': parse_cxid9114_cspad2()
    elif tname == '3': parse_xpptut15_pnccd()
    elif tname == '4': parse_xppl1316_opal1k()
    elif tname == '5': parse_xppn4116_opal1k()
    elif tname == '6': parse_mfxm5116_opal4k()
    elif tname == '7': parse_meca6013_opal8k()
    else : sys.exit('Test %s is not implemented' % tname)

    sys.exit('End of Test %s' % tname)

#--------------------

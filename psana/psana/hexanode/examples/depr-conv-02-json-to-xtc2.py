#!/usr/bin/env python

# run in LCLS2 environment
# python conv-02-json-to-xtc2.py [number]

#import numpy as np
#import dgramCreate as dc
import numpy as np
import dgramCreate as dc
import os, pickle, json, base64

from psana.dgrammanager import DgramManager
from psana import DataSource

#--------------------

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    event_dict = []
    for event in data:
        for key,val in event.items():
            try:
                event[key]= np.frombuffer(base64.b64decode(val[0]), dtype = np.dtype(val[2])).reshape(val[1])
            except TypeError:
                pass
        event_dict.append(event)
    return event_dict

#--------------------

def translate_xtc_demo(ifname, det_type='cspad', offset=1):
    xtcfile = '%s.xtc2' % ifname.rstrip('.json')
    print('Get data from file: %s' % ifname)
    print('Save data  in file: %s' % xtcfile)

    try:
        os.remove(xtcfile)
    except:
        pass

    lcls1_xtc = load_json(ifname)

    cfg_namesId = 0
    cfg_alg = dc.alg('cfg', [1, 2, 3])
    cfg_ninfo = dc.nameinfo('my'+det_type, det_type, 'serialnum1234', cfg_namesId)

    evt_namesId = 1
    evt_alg = dc.alg('raw', [4, 5, 6])
    evt_ninfo = dc.nameinfo('my'+det_type, det_type, 'serialnum1234', evt_namesId)

    # This is needed for some reason.
    # Perhaps a collision of lcls1 xtc "version" with lcls2 counterpart
    try:
        lcls1_xtc[0]['version.'] = lcls1_xtc[0]['version']
        del lcls1_xtc[0]['version']
    except KeyError:
        pass

    cydgram = dc.CyDgram()

    with open(xtcfile, 'wb') as f:
        # the order of these two lines must match the of the namesId's above
        cydgram.addDet(cfg_ninfo, cfg_alg, lcls1_xtc[0])
        cydgram.addDet(evt_ninfo, evt_alg, lcls1_xtc[offset])
        df = cydgram.get(0,0)
        f.write(df)

        # this currently duplicates the first dgram
        for event_dgram in lcls1_xtc[offset:]:
            cydgram.addDet(evt_ninfo, evt_alg, event_dgram)
            df = cydgram.get(0,0)
            f.write(df)

# Examples
# The LCLS1 dgrams are organized differently than the LCLS2 version
# In LCLS2, the configure contains all the names in the subsequent event dgrams
# This isn't the case for LCLS1, so I define the first event as a pseudo-configure
# and the second event as a configure for the rest of the events
# There is an edge case for the crystal data, as the second event is blank
# This causes a segfault when loading the LCLS2 xtc file with DgramManager
# My workaround is to define the second event as the configure for the events
# with the optional offset argument to translate_xtc_demo


#translate_xtc_demo('data-cxid9114-r0089-e000050-cspad.json')
#translate_xtc_demo('data-xpptut15-r0430-e000010-jungfrau.json')
#translate_xtc_demo('epix')
#translate_xtc_demo('crystal_dark', 2)
#translate_xtc_demo('crystal_xray', 2)

#--------------------

if __name__ == "__main__" :
    import sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'

    if   tname == '0': translate_xtc_demo('data-xpptut15-r0430-e000010-jungfrau.json', det_type='jungfrau')
    elif tname == '1': translate_xtc_demo('data-cxid9114-r0089-e000010-cspad.json', det_type='cspad')
    elif tname == '2': translate_xtc_demo('data-xpptut15-r0450-e000010-pnccd.json', det_type='pnccd')
    elif tname == '3': translate_xtc_demo('data-sxrx20915-r0206-e000010-pnccd.json', det_type='pnccd')
    elif tname == '4': translate_xtc_demo('data-xppl1316-r0193-e000010-opal1k.json', det_type='opal1k')
    elif tname == '5': translate_xtc_demo('data-xppn4116-r0137-e000010-opal1k.json', det_type='opal1k')
    elif tname == '6': translate_xtc_demo('data-mfxm5116-r0020-e000010-opal4k.json', det_type='opal4k')
    elif tname == '7': translate_xtc_demo('data-meca6013-r0009-e000010-opal8k.json', det_type='opal8k')
    else : sys.exit('Test %s is not implemented' % tname)

    sys.exit('End of Test %s' % tname)

#--------------------

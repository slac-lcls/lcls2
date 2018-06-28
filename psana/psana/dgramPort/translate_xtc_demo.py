import numpy as np
import dgramCreate as dc
import os, pickle, json, base64

from psana.dgrammanager import DgramManager
from psana import DataSource

import sys, os
outdir = ''
if len(sys.argv) == 2: outdir = sys.argv[1]

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


def translate_xtc_demo(job_type, offset=1):
    configure_file = '%s_configure.xtc' % job_type
    event_file = '%s_evts.xtc' % job_type

    try:
        os.remove(FILE_NAME)
    except:
        pass

    lcls1_xtc = load_json("%s.json" % job_type)

    alg = dc.alg('raw', [1, 2, 3])
    ninfo = dc.nameinfo('DsdCsPad', 'cspad', 'detnum1234', 0)

    # This is needed for some reason.
    # Perhaps a collision of lcls1 xtc "version" with lcls2 counterpart
    try:
        lcls1_xtc[0]['version.'] = lcls1_xtc[0]['version']
        del lcls1_xtc[0]['version']
    except KeyError:
        pass

    cydgram = dc.CyDgram()

    with open(configure_file, 'wb') as f:
        cydgram.addDet(ninfo, alg, lcls1_xtc[0])
        df = cydgram.get()
        f.write(df)

    del cydgram
    cydgram = dc.CyDgram()

    with open(event_file, 'wb') as f:
        for event_dgram in lcls1_xtc[offset:]:
            cydgram.addDet(ninfo, alg, event_dgram)
            df = cydgram.get()
            f.write(df)

    ds_cfg = DgramManager(configure_file)
    ds_evt = DgramManager(event_file)

# Examples
# The LCLS1 dgrams are organized differently than the LCLS2 version
# In LCLS2, the configure contains all the names in the subsequent event dgrams
# This isn't the case for LCLS1, so I define the first event as a pseudo-configure
# and the second event as a configure for the rest of the events
# There is an edge case for the crystal data, as the second event is blank
# This causes a segfault when loading the LCLS2 xtc file with DgramManager
# My workaround is to define the second event as the configure for the events
# with the optional offset argument to translate_xtc_demo


#translate_xtc_demo(os.path.join(outdir,'jungfrau'))
#translate_xtc_demo(os.path.join(outdir,'epix'))
translate_xtc_demo(os.path.join(outdir,'crystal_dark'), 2)
translate_xtc_demo(os.path.join(outdir,'crystal_xray'), 2)
import numpy as np
import dgramCreate as dc
import os, pickle, json, base64

from psana.dgrammanager import DgramManager
from psana import DataSource


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


def translate_xtc_demo(job_type):
    configure_file = '%s_configure.xtc' % job_type
    event_file = '%s_evts.xtc' % job_type

    try:
        os.remove(FILE_NAME)
    except:
        pass

    lcls1_xtc = load_json("%s.json" % job_type)


    alg = dc.alg('alg', [0, 0, 0])

    ninfo = dc.nameinfo('lcls1', 'epix', 'detnum1234', 0)

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
        for event_dgram in lcls1_xtc[1:]:
            cydgram.addDet(ninfo, alg, event_dgram)
            df = cydgram.get()
            f.write(df)


    ds_cfg = DgramManager(configure_file)
    ds_evt = DgramManager(event_file)


# Examples

translate_xtc_demo('jungfrau')
translate_xtc_demo('epix')

translate_xtc_demo('crystal_dark')
translate_xtc_demo('crystal_xray')

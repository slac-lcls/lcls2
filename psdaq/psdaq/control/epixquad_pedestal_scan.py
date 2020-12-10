from psdaq.control.config_scan import scan
from psdaq.configdb.get_config import *
import numpy as np

detName  = 'epixquad_0'

def steps():
    d = {}
    metad = {}
    metad['detname'] = detName
    metad['scantype'] = 'pedestal'
    for gain in range(5):
        #  Set the detector level config change
        d[f'{detName}:user.gain_mode'] = gain
        #  Set the global meta data
        metad['step'] = gain
        yield (d, gain, json.dumps(metad))

if __name__ == '__main__':

    keys = []
    keys.append(f'{detName}:user.gain_mode')

    scan(keys, steps)

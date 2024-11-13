from psdaq.control.config_scan import scan
from psdaq.configdb.get_config import *
import numpy as np
import sys

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
        yield (d, float(gain), json.dumps(metad))

if __name__ == '__main__':

    # default command line arguments
    defargs = {'-B':'DAQ:UED',
               '-p':'0',
               '-x':'0',
               '-g':'1',
               '-C':'drp-ued-cmp002',
               '-c':'1000',
               '-X':'drp-ued-cmp001',
               '--config':'BEAM'}

    keys = []
    keys.append(f'{detName}:user.gain_mode')

    scan(keys, steps, defargs=defargs)

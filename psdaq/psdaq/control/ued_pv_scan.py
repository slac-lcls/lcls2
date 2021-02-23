from psdaq.control.config_scan import scan
from psdaq.configdb.get_config import *
import numpy as np
import argparse
import sys
import epics
import time
import math

class CaPutProc():
    def __init__(self, pvname, pvrbname, eps=0.0001):
        self.pvname = pvname
        self.pv  = epics.PV(pvname)
        self.pvm = epics.PV(pvrbname)
        self.eps = eps 

    def __call__(self, value):
        r = self.pv.put(value)
        print('put {}'.format(value))
        while True:
            v = self.pvm.get()
            print('  got {}'.format(v))
            if math.fabs(v - value) < self.eps:
                break
            time.sleep(0.05)
        return r

#
#  This generator function determines the PV steps
#
#def linear_steps():
#    d = {}
#    stepnum = 0
#    for value in np.arange(0.2,0.3,0.02):
#        d[motorName] = value
#        yield (d, stepnum, json.dumps(d))
#        #yield (d, value, json.dumps(d))
#        stepnum += 1

#
#  This function is executed before each DAQ step
#
#def setupStep(step):
#    caput(step[0][motorName])
#    #caput(step[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument('--pv', type=str, required=True, help='control PV')
    parser.add_argument('--pv', default='MOTR:AS01:MC05:CH6:MOTOR', help='control PV')
    parser.add_argument('--linear', type=float, nargs=3, help='linear scan over range [0]:[1] in steps of [2]')
    parser.add_argument('--events', default=2000, help='events per step (default 2000)')
    args = parser.parse_args()
    
    # command line arguments for config_scan.py:scan()
    del sys.argv[1:]
    if len(sys.argv)==1:
        defargs = ['-B','DAQ:UED','-p','0','-x','0','-C','drp-ued-cmp002',f'-c {args.events}','--config','BEAM']
        sys.argv.extend(defargs)

    keys = []
    keys.append(f'{args.pv}')

    if hasattr(args,'linear'):
        def steps():
            d = {}
            stepnum = 0
            for value in np.arange(*args.linear):
                d[args.pv] = value
                #yield (d, stepnum, json.dumps(d))
                yield (d, value, json.dumps(d))
                stepnum += 1

        caput = CaPutProc(args.pv,args.pv+'.RBV')

        def setupStep(step):
            caput(step[0][args.pv])
            #caput(step[1])

    else:
        raise RuntimeError('Must specify scan type (--linear,)')

    scan(keys, steps, setupStep)

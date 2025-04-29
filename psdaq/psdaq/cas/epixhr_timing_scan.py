from psdaq.cas.config_scan_base import ConfigScanBase
import numpy as np
import json

def main():

    # default command line arguments
    defargs = {'--events'  :1000,
               '--hutch'   :'rix',
               '--detname' :'epixhr_0',
               '--scantype':'timing',
               '--record'  :1}

    aargs = [('--linear',{'type':float,'nargs':3,'help':'linear scan over range [0]:[1] in steps of [2]'})]
    scan = ConfigScanBase(userargs=aargs, defargs=defargs)
    args = scan.args

    keys = [f'{args.detname}:user.start_ns']

    # scan loop
    aargs = args.linear if args.linear else (107700.,1107700.,100000.)
    d = {}
    metad = {'detname':args.detname,
             'scantype':args.scantype}
    def steps():
        for i,value in enumerate(np.arange(*aargs)):
            d[f'{args.detname}:user.start_ns'] = int(value)
            metad['step'] = value
            yield (d, value, json.dumps(metad))

    scan.run(keys,steps)

if __name__ == '__main__':
    main()

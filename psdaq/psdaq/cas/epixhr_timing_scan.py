from psdaq.cas.config_scan_base import ConfigScanBase
import numpy as np
import json

def main():

    aargs = [('--linear',{'type':float,'nargs':3,'help':'linear scan over range [0]:[1] in steps of [2]'})]
    scan = ConfigScanBase(aargs)

    args = scan.args

    keys = [f'{args.detname}:user.start_ns']

    # scan loop
    aargs = args.linear if args.linear else (107700.,1107700.,100000.)
    d = {}
    metad = {'detname':args.detname,
             'scantype':'timing'}
    def steps():
        for i,value in enumerate(np.arange(*aargs)):
            d['f{args.detname}:user.start_ns'] = value
            metad['step'] = i
            yield (d, value, json.dumps(metad))

    scan.run(keys,steps)

if __name__ == '__main__':
    main()

from psdaq.cas.config_scan_base import ConfigScanBase
import numpy as np
import json
    
def main():

    aargs = [('--linear',{'type':float,'nargs':3,'help':'linear scan over range [0]:[1] in steps of [2]'})]

    scan = ConfigScanBase(aargs)
             
    args = scan.args

    keys = [f'{args.detname}:expert.EpixHR.RegisterControl.R0Delay',
            f'{args.detname}:expert.EpixHR.RegisterControl.AcqDelay1']

    d = {}
    for i,k in enumerate(keys):
        d[k] = 0

    def steps():
        metad = {'detname':args.detname, 'scantype':args.scantype}
        d = {}
        for value in np.arange(0.,args.step*args.nsteps,args.step):
            for i,k in enumerate(keys):
                d[k] = int(value+args.start[i])
            yield (d, value, json.dumps(metad))

    if args.linear:
        print(f'linear: {args.linear}')
        def steps():
            metad = {'detname':args.detname, 'scantype':args.scantype}
            d = {}
            for value in np.arange(*args.linear):
                for k in keys:
                    d[k] = int(value)
                yield (d, value, json.dumps(metad))

    else:
        raise RuntimeError('Must specify scan type (--linear,)')


    scan.run(keys,steps)

if __name__ == '__main__':
    main()

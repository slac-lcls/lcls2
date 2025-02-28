from psdaq.cas.config_scan_base import ConfigScanBase
import json

def main():

    # default command line arguments
    defargs = {'--hutch'   :'rix',
               '--detname' :'epixhr_0',
               '--scantype':'pedestal',
               '--events'  :1000,
               '--record'  :1}

    scan = ConfigScanBase(defargs=defargs)
    args = scan.args

    args.scantype = 'pedestal'
    keys = [f'{args.detname}:user.gain_mode']

    def steps():
        d = {}
        metad = {'detname':args.detname,
                 'scantype':args.scantype}
        for gain in range(5):
            d[f'{args.detname}:user.gain_mode'] = int(gain)
            metad['step'] = int(gain)
            yield (d, float(gain), json.dumps(metad))

    scan.run(keys,steps)

if __name__ == '__main__':
    main()

from psdaq.cas.config_scan_base import ConfigScanBase
from psdaq.configdb.epixm320_utils import *
import json

def main():

    # default command line arguments
    defargs = {'--events'  :1000,
               '--hutch'   :'rix',
               '--detname' :'epixm320_0',
               '--scantype':'pedestal',
               '--record'  :1}

    scan = ConfigScanBase(defargs=defargs)

    args = scan.args

    keys = [f'{args.detname}:user.gain_mode']

    def steps():
        d = {}
        metad = {'detname':args.detname,
                 'scantype':args.scantype}
        for gain_mode in ('AHL', 'SH'):
            gain = gain_mode_value(gain_mode)
            d[f'{args.detname}:user.gain_mode'] = int(gain)
            metad['gain_mode'] = gain_mode
            metad['step'] = int(gain)
            yield (d, float(gain), json.dumps(metad))

    scan.run(keys,steps)

if __name__ == '__main__':
    main()

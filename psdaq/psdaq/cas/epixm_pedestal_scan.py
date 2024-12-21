from psdaq.cas.config_scan_base import ConfigScanBase
from psdaq.configdb.epixm320_config import gain_mode_value
import json

def main():

    # default command line arguments
    defargs = {'--events'  :1000,
               '--hutch'   :'rix',
               '--detname' :'epixm_0',
               '--scantype':'pedestal',
               '--record'  :1}

    scan = ConfigScanBase(defargs=defargs)

    args = scan.args

    keys = [f'{args.detname}:user.gain_mode']

    def steps():
        d = {}
        metad = {'detname' : args.detname,
                 'scantype': args.scantype,
                 'events'  : args.events}
        step = 0
        for gain_mode in ('AHL', 'SH'):
            gain = gain_mode_value(gain_mode)
            d[f'{args.detname}:user.gain_mode'] = int(gain)
            metad['gain_mode'] = gain_mode
            metad['step']      = int(gain)
            yield (d, float(step), json.dumps(metad))
            step += 1

    scan.run(keys,steps)

if __name__ == '__main__':
    main()

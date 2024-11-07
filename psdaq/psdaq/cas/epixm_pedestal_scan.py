from psdaq.cas.config_scan_base import ConfigScanBase
from psdaq.configdb.epixm320_utils import *
import json

def main():

    scan = ConfigScanBase(scantype='pedestal')

    args = scan.args
    args.scantype = 'pedestal'
    keys = [f'{args.detname}:user.gain_mode']

    def steps():
        d = {}
        metad = {'detname':args.detname,
                 'scantype':args.scantype}
        for gain in range(3):
            d[f'{args.detname}:user.gain_mode'] = int(gain)
            metad['gain_mode'] = gain_mode_map(gain)[2]
            metad['step'] = int(gain)
            yield (d, float(gain), json.dumps(metad))

    scan.run(keys,steps)

if __name__ == '__main__':
    main()

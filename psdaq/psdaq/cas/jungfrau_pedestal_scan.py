from psdaq.cas.config_scan_base import ConfigScanBase
import json

def main():

    # default command line arguments
    defaults = {
        "--events": 1000,
        "--hutch": "asc",
        "--detname": "jungfrau_0",
        "--scantype": "pedestal",
        "--record": 0,#1,
        "--config": "BEAM",
    }

    scan = ConfigScanBase(defargs=defaults)
    args = scan.args

    keys = []
    keys.append(f'{args.detname}:user.gainMode')

    def steps():
        d = {}
        metad = {}
        metad['detname'] = args.detname
        metad['scantype'] = 'pedestal'
        for gain in range(5):
            #  Set the detector level config change
            d[f'{args.detname}:user.gainMode'] = gain
            #  Set the global meta data
            metad['step'] = gain
            yield (d, float(gain), json.dumps(metad))

    scan.run(keys, steps)

if __name__ == '__main__':
    main()

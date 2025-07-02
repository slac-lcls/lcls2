from psdaq.cas.config_scan_base import ConfigScanBase
import json

def main():

    # default command line arguments
    defaults = {
        "--events": 1000,
        "--hutch": "asc",
        "--detname": "jungfrau",
        "--scantype": "pedestal",
        "--record": 1,
        "--config": "BEAM",
        "--nprocs": 5,
    }

    scan = ConfigScanBase(defargs=defaults)
    args = scan.args

    keys = []
    for i in range(args.nprocs):
        keys.append(f'{args.detname}_{i}:user.gainMode')

    def steps():
        d = {}
        metad = {}
        metad['detname'] = args.detname
        metad['scantype'] = 'pedestal'
        for gain in range(3):
            #  Set the detector level config change
            for i in range(args.nprocs):
                d[f'{args.detname}_{i}:user.gainMode'] = gain
            #  Set the global meta data
            metad['step'] = gain
            metad['gainMode'] = gain
            yield (d, float(gain), json.dumps(metad))

    scan.run(keys, steps)

if __name__ == '__main__':
    main()

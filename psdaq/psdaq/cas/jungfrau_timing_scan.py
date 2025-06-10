from psdaq.cas.config_scan_base import ConfigScanBase
import json
import numpy as np

def main():

    # default command line arguments
    defaults = {
        "--events": 1000,
        "--hutch": "mfx",
        "--detname": "jungfrau",
        "--scantype": "timing",
        "--record": 1,
        "--config": "BEAM",
        "--nprocs": 5,
    }

    scan = ConfigScanBase(defargs=defaults)
    args = scan.args

    keys = []
    for i in range(args.nprocs):
        keys.append(f'{args.detname}_{i}:user.trigger_delay_s')

    def steps():
        d = {}
        metad = {}
        metad['detname'] = args.detname
        metad['scantype'] = 'timing'
        timing_vals = np.linspace(0.000200,0.000220, 20)
        for delay_idx in range(len(timing_vals)):
            #  Set the detector level config change
            for i in range(args.nprocs):
                d[f'{args.detname}_{i}:user.trigger_delay_s'] = timing_vals[delay_idx]
            #  Set the global meta data
            metad['step'] = delay_idx
            metad['trigger_delay_s'] = timing_vals[delay_idx]
            yield (d, float(delay_idx), json.dumps(metad))

    scan.run(keys, steps)

if __name__ == '__main__':
    main()

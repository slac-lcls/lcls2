from psdaq.cas.config_scan_base import ConfigScanBase
import json
import numpy as np

def main():

    # default command line arguments
    defaults = {
        "--events": 1000,
        "--hutch": "mfx",
        "--detname": "jungfrau_0",
        "--scantype": "timing",
        "--record": 0,#1,
        "--config": "BEAM",
    }

    scan = ConfigScanBase(defargs=defaults)
    args = scan.args

    keys = []
    keys.append(f'{args.detname}:user.trigger_delay_s')

    def steps():
        d = {}
        metad = {}
        metad['detname'] = args.detname
        metad['scantype'] = 'timing'
        timing_vals = np.linspace(0.000200,0.000220, 20)
        for delay_idx in range(20):
            #  Set the detector level config change
            d[f'{args.detname}:user.trigger_delay_s'] = timing_vals[delay_idx]
            #  Set the global meta data
            metad['step'] = delay_idx
            metad['trigger_delay_s'] = timing_vals[delay_idx]
            yield (d, float(delay_idx), json.dumps(metad))

    scan.run(keys, steps)

if __name__ == '__main__':
    main()

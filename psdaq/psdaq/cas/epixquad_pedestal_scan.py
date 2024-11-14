from psdaq.cas.config_scan_base import ConfigScanBase
import json

def main():

    # default command line arguments
    defargs = {'-c'        :1000,
               '--hutch'   :'ued',
               '--detname' :'epixquad_0',
               '--scantype':'pedestal',
               '--record'  :1}

    scan = ConfigScanBase(defargs=defargs)
    args = scan.args

    keys = []
    keys.append(f'{args.detname}:user.gain_mode')

    def steps():
        d = {}
        metad = {}
        metad['detname'] = args.detname
        metad['scantype'] = 'pedestal'
        for gain in range(5):
            #  Set the detector level config change
            d[f'{args.detname}:user.gain_mode'] = gain
            #  Set the global meta data
            metad['step'] = gain
            yield (d, float(gain), json.dumps(metad))

    scan.run(keys, steps)

if __name__ == '__main__':
    main()


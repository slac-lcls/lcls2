from psdaq.cas.config_scan_base import ConfigScanBase
import json

def main():

    scan = ConfigScanBase()

    args = scan.args
    args.scantype = 'pedestal'
    keys = [f'{args.detname}:user.gain_mode']

    def steps():
        d = {}
        metad = {'detname':args.detname,
                 'scantype':'pedestal'}
        for gain in range(5):
            d[f'{args.detname}:user.gain_mode'] = int(gain)
            metad['step'] = int(gain)
            yield (d, float(gain), json.dumps(metad))

    scan.run(keys,steps)

if __name__ == '__main__':
    main()

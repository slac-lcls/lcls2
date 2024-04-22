from psdaq.cas.config_scan_base import ConfigScanBase
import numpy as np
import json

def main():

    aargs = [('--spacing',{'type':int,'default':5,'help':'size of rectangular grid'})]
    scan = ConfigScanBase(aargs, scantype='chargeinj')

    args = scan.args
    keys = []
    keys.append(f'{args.detname}:user.gain_mode')
    keys.append(f'{args.detname}:user.pixel_map')
    for a in range(4):
        saci = f'{args.detname}:expert.EpixHR.Hr10kTAsic{a}'
        keys.append(f'{saci}.atest')
        keys.append(f'{saci}.test' )
        keys.append(f'{saci}.trbit' )
        keys.append(f'{saci}.Pulser')

    # scan loop
    spacing  = args.spacing

    def pixel_mask(value0,value1,spacing,position):
        ny,nx=288,384;
        if position>=spacing**2:
            print('position out of range')
            position=0;
        #    print 'pixel_mask(', value0, value1, spacing, position, ')'
        out=np.zeros((ny,nx),dtype=np.uint8)+value0;
        position_x=position%spacing; position_y=position//spacing;
        out[position_y::spacing,position_x::spacing]=value1;
        return out

    def steps():
        d = {}
        metad = {}
        metad['detname'] = args.detname
        metad['scantype'] = args.scantype
        d[f'{args.detname}:user.gain_mode'] = 5  # Map
        for a in range(4):
            saci = f'{args.detname}:expert.EpixHR.Hr10kTAsic{a}'
            d[f'{saci}.atest'] = 1
            d[f'{saci}.test' ] = 1
            d[f'{saci}.Pulser'] = 0xc8
            # d[f'{saci}:PulserSync'] = 1  # with ghost correction
        for trbit in [0,1]:
            for a in range(4):
                saci = f'{args.detname}:expert.EpixHR.Hr10kTAsic{a}'
                d[f'{saci}.trbit'] = trbit
            for s in range(spacing**2):
                pmask = pixel_mask(0,1,spacing,s)
                #  Do I need to convert to list and lose the dimensionality? (json not serializable)
                d[f'{args.detname}:user.pixel_map'] = pmask.reshape(-1).tolist()
                #d[f'{args.detname}:user.pixel_map'] = pmask
                #  Set the global meta data
                metad['step'] = s+trbit*spacing**2

                yield (d, float(s+trbit*spacing**2), json.dumps(metad))

    scan.run(keys,steps)

if __name__ == '__main__':
    main()

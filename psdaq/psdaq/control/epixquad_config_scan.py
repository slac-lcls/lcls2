from psdaq.control.config_scan import scan
from psdaq.configdb.get_config import *
import numpy as np

getDarks = True  # or charge injection
spacing  = 2
detName  = 'epixquad_0'

def pixel_mask(value0,value1,spacing,position):
    ny,nx=352,384;
    if position>=spacing**2:
        print('position out of range')
        position=0;
#    print 'pixel_mask(', value0, value1, spacing, position, ')'
    out=np.zeros((ny,nx),dtype=np.int)+value0;
    position_x=position%spacing; position_y=position//spacing;
    out[position_y::spacing,position_x::spacing]=value1;
    p = np.pad(out,[(0,4),(0,0)],mode='constant')
    a = np.zeros((4,178,192),dtype=np.uint8)
    a[0] = p[:178,:192]
    a[1] = p[:178,192:]
    a[2] = p[178:,:192]
    a[3] = p[178:,192:]
    quad = np.zeros((16,178,192),dtype=np.uint8)
    for i in range(4):
        quad[i::4] = a[i]
    return quad;

def steps():
    d = {}
    metad = {}
    metad['detname'] = detName
    if getDarks:
        #  Darks
        metad['scantype'] = 'pedestal'
        for gain in range(5):
            #  Set the detector level config change
            d[f'{detName}:user.gain_mode'] = gain
            #  Set the global meta data
            metad['step'] = gain
            yield (d, gain, json.dumps(metad))
    else:
        #
        #  Charge Injection
        #
        metad['scantype'] = 'chargeinj'
        d[f'{detName}:user.gain_mode'] = 5  # Map
        for a in range(16):
            saci = f'{detName}:expert.EpixQuad.Epix10kaSaci[{a}]'
            d[f'{saci}.atest'] = 1
            d[f'{saci}.test' ] = 1
            d[f'{saci}.Pulser'] = 0xc8
            # d[f'{saci}:PulserSync'] = 1  # with ghost correction
        for trbit in [0,1]:
            for a in range(16):
                saci = f'{detName}:expert.EpixQuad.Epix10kaSaci[{a}]'
                d[f'{saci}.trbit'] = trbit
            for s in range(spacing**2):
                pmask = pixel_mask(0,1,spacing,s)
                d[f'{detName}:user.pixel_map'] = pmask.reshape(-1).tolist()
                #  Set the global meta data
                metad['step'] = s+trbit+spacing**2

                yield (d, s+trbit+spacing**2, json.dumps(metad))

if __name__ == '__main__':

    keys = []
    if getDarks:
        keys.append(f'{detName}:user.gain_mode')
    else:
        keys.append(f'{detName}:user.gain_mode')
        keys.append(f'{detName}:user.pixel_map')
        for a in range(16):
            saci = f'{detName}:expert.EpixQuad.Epix10kaSaci[{a}]'
            keys.append(f'{saci}.atest')
            keys.append(f'{saci}.test' )
            keys.append(f'{saci}.trbit' )
            keys.append(f'{saci}.Pulser')

#    print(f'keys {keys}')
#    print('steps:')
#    for s in steps():
#        print(s)
    scan(keys, steps)

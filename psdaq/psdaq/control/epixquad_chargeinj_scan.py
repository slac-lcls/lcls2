from psdaq.control.config_scan import scan
from psdaq.configdb.get_config import *
import numpy as np
import sys

spacing  = 7
detName  = 'epixquad_0'

def pixel_mask(value0,value1,spacing,position_x,position_y):
    ny,nx=352,384;
    out=np.zeros((ny,nx),dtype=np.int)+value0;
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
            position_x=position%spacing
            position_y=position//spacing;
            pmask = pixel_mask(0,1,spacing,position_x,position_y)
            d[f'{detName}:user.pixel_map'] = pmask.reshape(-1).tolist()
            #  Set the global meta data
            metad['step'] = s+trbit*spacing**2
            metad['injpos'] = [position_y,position_x]

            yield (d, float(s+trbit*spacing**2), json.dumps(metad))

if __name__ == '__main__':

    # default command line arguments
    if len(sys.argv)==1:
        defargs = ['-B','DAQ:UED','-p','0','-x','0','-C','drp-ued-cmp002','-c 2000','--config','BEAM']
        sys.argv.extend(defargs)

    keys = []
    keys.append(f'{detName}:user.gain_mode')
    keys.append(f'{detName}:user.pixel_map')
    for a in range(16):
        saci = f'{detName}:expert.EpixQuad.Epix10kaSaci[{a}]'
        keys.append(f'{saci}.atest')
        keys.append(f'{saci}.test' )
        keys.append(f'{saci}.trbit' )
        keys.append(f'{saci}.Pulser')

    scan(keys, steps)

from psdaq.configdb.epixquad_cdict import epixquad_cdict
import psdaq.configdb.configdb as cdb
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

def copyValues(din,dout,k=None):
    if k is not None and ':RO' in k:
        return
    if isinstance(din,dict):
        for key,value in din.items():
            copyValues(value,dout,key if k is None else k+'.'+key)
    else:
        v = dout.get(k,withtype=True)
        if v is None:
            pass
        elif len(v)>2:
            print(f'Skipping {k}')
        elif len(v)==1:
            print(f'Updating {k}')
            dout.set(k,din,'UINT8')
        else:
            print(f'Updating {k}')
            dout.set(k,din,v[0])

def epixquad_roi(xroi,yroi,inner,outer):
    d = {}
    gain_to_value = (0xc,0xc,0x8,0x0,0x0) # H/M/L/AHL/AML
    gain_to_trbit = (0x1,0x0,0x0,0x1,0x0) # H/M/L/AHL/AML

    if gain_to_trbit[inner]!=gain_to_trbit[outer]:
        raise ValueError(f'Incompatible gains {inner},{outer} for pixel configuration')

    trbit = gain_to_trbit[inner]
    for i in range(16):
        d[f'expert.EpixQuad.Epix10kaSaci{i}.trbit'] = trbit

    pixelMap = np.zeros((16,178,192),dtype=np.uint8)+gain_to_value[outer]
    image    = np.zeros(((4*178),(4*192)),dtype=np.uint8)+gain_to_value[outer]

    vinner = gain_to_value[inner]
    
    #  Copy some code from lcls1 pdsapp/config/Epix10kaQuadGainMap.cc:
    shape = (4,352,384)
    _height = 176
    _width  = 192
    _xoff   = 4
    _yoff   = 4
    #asic_xo = ( _xoff, _xoff           , _width +_xoff+1  , _width+_xoff+1 )
    #asic_yo = ( _yoff, _height+_yoff+1 , _height+_yoff+1  , _yoff )
    #elem_xo = ( 0    , 2*_width+_xoff+2, 0                , 2*_width+_xoff+2 )
    #elem_yo = ( 0    , 0               , 2*_height+_xoff+2, 2*_height+_xoff+2 )
    asic_xo = ( _width +_xoff+1, _width+_xoff+1, _xoff, _xoff )
    asic_yo = ( _height+_yoff+1, _yoff         , _yoff, _height+_yoff+1 )
    elem_xo = ( 2*_width+_xoff+2, 0, 2*_width+_xoff+2 , 0 )
    elem_yo = ( 0               , 0, 2*_height+_xoff+2, 2*_height+_xoff+2 )

    #  Loop over asics
    for j in range(16):
        ie = int(j/4)  # element
        ia = j%4  # asic
        xo = elem_xo[ie] + asic_xo[ia]
        yo = elem_yo[ie] + asic_yo[ia]

        sh = (352,384)
        xoff = sh[1]-1 if (ia==0 or ia==1) else sh[1]/2-1
        yoff = sh[0]-1 if (ia==0 or ia==3) else sh[0]/2-1

        #  Loop over pixels in asic
        for y in range(0,176):
            for x in range(0,192):
                #  Are we in the ROI?
                if (xo+x > xroi[0] and xo+x < xroi[1] and
                    yo+y > yroi[0] and yo+y < yroi[1]):
                    #pixelMap[j][yoff-y][xoff-x] = vinner
                    pixelMap[j][y][x] = vinner
                    image[xo+x][yo+y] = vinner
                if x%32==0 and y%32==0:
                    print(f'ax {x}  ay {y}  xo+x {xo+x}  yo+y {yo+y}  xroi {xroi}  yroi {yroi}')

    vouter = gain_to_value[outer]
    inner_pixels = pixelMap - vouter
    outer_pixels = pixelMap - vinner
    print(f'Inner region entries {np.count_nonzero(inner_pixels)}')
    print(f'Outer region entries {np.count_nonzero(outer_pixels)}')

    d['user.pixel_map'] = pixelMap
    return (d,image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update gain map')
    parser.add_argument('--x', help='low gain x-bounds', type=int, nargs=2, required=True)
    parser.add_argument('--y', help='low gain x-bounds', type=int, nargs=2, required=True)

    parser.add_argument('--dev', help='use development db', action='store_true')
    parser.add_argument('--inst', help='instrument', type=str, default='tmo')
    parser.add_argument('--alias', help='alias name', type=str, default='BEAM')
    parser.add_argument('--name', help='detector name', type=str, default='tmoopal2')
    parser.add_argument('--segm', help='detector segment', type=int, default=0)
    parser.add_argument('--id', help='device id/serial num', type=str, default='serial1234')
    parser.add_argument('--user', help='user for HTTP authentication', type=str, default='tstopr')
    parser.add_argument('--password', help='password for HTTP authentication', type=str, default='pcds')
    args = parser.parse_args()

    create = False
    dbname = 'configDB'

    detname = f'{args.name}_{args.segm}'

    db   = 'devconfigdb' if args.dev else 'configdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    create = False

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    cfg = mycdb.get_configuration(args.alias,detname)

    if cfg is None: raise ValueError('Config for instrument/detname %s/%s not found. dbase url: %s, db_name: %s, config_style: %s'%(args.inst,detname,url,dbname,args.alias))

    top = epixquad_cdict()  # We need the full cdict in order to store
    copyValues(cfg,top)     # Now copy old values into the cdict

    #  Write gainmap
    d,image = epixquad_roi(args.x,args.y,2,1)
    copyValues(d,top)

    print('Setting user.pixel_map')
    top.set('user.pixel_map', d['user.pixel_map'])

    #  Store
    top.setInfo('epix10kaquad', args.name, args.segm, args.id, 'No comment')
    mycdb.modify_device(args.alias, top)

    #  Show image
    print(d['user.pixel_map'])
    plt.imshow(image)
    plt.show()

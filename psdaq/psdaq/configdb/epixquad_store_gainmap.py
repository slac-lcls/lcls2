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

    _height = 192
    _width  = 176
    pixelMap = np.zeros((16,_width+2,_height),dtype=np.uint8)+gain_to_value[outer]
    image    = np.zeros(((4*_width),(4*_height)),dtype=np.uint8)+gain_to_value[outer]

    vinner = gain_to_value[inner]

    #  Approximate the geometry
    _xoff   = 4
    _yoff   = 4
    asic_xo = ( _xoff, _width+_xoff+1 , _width+_xoff+1  , _xoff )
    asic_yo = ( _yoff, _yoff          , _height +_yoff+1, _height+_yoff+1 )
    elem_xo = ( 2*_width +_xoff+2, 2*_width+_xoff+2, 0                , 0 )
    elem_yo = ( 2*_height+_yoff+2, 0               , 2*_height+_yoff+2, 0 )

    #  Loop over asics
    for j in range(16):
        ie = int(j/4)  # element
        ia = j%4  # asic
        xo = elem_xo[ie] + asic_xo[ia]
        yo = elem_yo[ie] + asic_yo[ia]

        sh = (352,384)
        xoff = _width-1
        yoff = _height-1

        #  Loop over pixels in asic
        for x in range(0,_width):
            for y in range(0,_height):
                #  Are we in the ROI?
                if (xo+x > xroi[0] and xo+x < xroi[1] and
                    yo+y > yroi[0] and yo+y < yroi[1]):
                    pixelMap[j][xoff-x][yoff-y] = vinner
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
    parser.add_argument('--inst', help='instrument', type=str, default='ued')
    parser.add_argument('--alias', help='alias name', type=str, default='BEAM')
    parser.add_argument('--name', help='detector name', type=str, default='epixquad')
    parser.add_argument('--segm', help='detector segment', type=int, default=0)
    parser.add_argument('--id', help='device id/serial num', type=str, default='serial1234')
    parser.add_argument('--user', help='user for HTTP authentication', type=str, default='uedopr')
    parser.add_argument('--password', help='password for HTTP authentication', type=str, default='pcds')
    parser.add_argument('--test', help='test transformation', action='store_true')
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
    if not args.test:
        mycdb.modify_device(args.alias, top)

    #  Show image
    print(d['user.pixel_map'])
#    plt.imshow(np.rot90(image))
    plt.pcolormesh(image.transpose())
    plt.show()

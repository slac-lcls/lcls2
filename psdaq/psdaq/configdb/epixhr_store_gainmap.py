from psdaq.configdb.epixhr2x2_config_store import epixhr2x2_cdict,elemRows,elemCols
from psdaq.configdb.typed_json import updateValue
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
            updateValue(dout,k,din)
#            dout.set(k,din,v[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update gain map')
    parser.add_argument('--trbit', help='trbit setting', type=int, default=0)
    parser.add_argument('--pixel', help='pixel setting', type=int, default=0)
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

    #  Set gainmap
    for i in range(4):
        cfg['expert']['EpixHR'][f'Hr10kTAsic{i}']['trbit'] = args.trbit
    pixelMap = np.zeros((elemRows*2,elemCols*2),dtype=np.uint8)+args.pixel

    print(cfg)

    top = epixhr2x2_cdict()  # We need the full cdict in order to store
    copyValues(cfg,top)     # Now copy old values into the cdict

    print('Setting user.pixel_map')
    top.set('user.pixel_map', pixelMap)

    #  Store
    top.setInfo('epixhr2x2hw', args.name, args.segm, args.id, 'No comment')
    if not args.test:
        mycdb.modify_device(args.alias, top)


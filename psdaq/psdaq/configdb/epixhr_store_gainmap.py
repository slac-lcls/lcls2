import os

from psdaq.configdb.epixhr2x2_config_store import epixhr2x2_cdict,elemRows,elemCols
from psdaq.configdb.typed_json import updateValue
import psdaq.configdb.configdb as cdb
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

elemRows = 144
elemCols = 192

gain_to_value = (0xc,0xc,0x8,0x0,0x0) # H/M/L/AHL/AML
gain_to_trbit = (0x1,0x0,0x0,0x1,0x0) # H/M/L/AHL/AML
gain_dict = {'H':{'trbit':1,'value':0xc},
             'M':{'trbit':0,'value':0xc},
             'L':{'trbit':0,'value':0x8},
             'AHL':{'trbit':1,'value':0x0},
             'AML':{'trbit':0,'value':0x0}}

def copyValues(din,dout,k=None):
    if isinstance(din,dict) and isinstance(dout[k],dict):
        for key,value in din.items():
            if key in dout[k]:
                copyValues(value,dout[k],key)
            else:
                print(f'skip {key}')
    elif isinstance(din,bool):
        vin = 1 if din else 0
        if dout[k] != vin:
            print(f'Writing {k} = {vin}')
            dout[k] = 1 if din else 0
        else:
            print(f'{k} unchanged')
    else:
        if dout[k] != din:
            print(f'Writing {k} = {din}')
            dout[k] = din
        else:
            print(f'{k} unchanged')

def epixhr_readmap(fname,gains):
    d = {'expert':{'EpixHR':{}}}

    if gain_dict[gains[0]]['trbit']!=gain_dict[gains[1]]['trbit']:
        raise ValueError(f'Incompatible gains {gains} for pixel configuration')

    trbit = gain_dict[gains[0]]['trbit']
    for i in range(4):
        d['expert']['EpixHR'][f'Hr10kTAsic{i}'] = {'trbit':trbit}

    vgain0 = gain_dict[gains[0]]['value']
    vgain1 = gain_dict[gains[1]]['value']

    e = np.genfromtxt(fname,dtype=np.uint8)
    print(f'genfromtxt {e.size}')
    e = e*(vgain1-vgain0) + vgain0
    
    #  break the elements into asics
    if False:
        e = np.vsplit(e,4)
        print(f'vsplit {len(e)} {e[0].size*4}')
        pca = []
        for i in e:
            a = []
            for j in np.vsplit(i,2):
                a.extend(np.hsplit(j,2))
            #pca.extend([np.asarray(a[3],dtype=np.uint8),
            #            np.flipud(np.fliplr(a[0])),
            #            np.flipud(np.fliplr(a[1])),
            #            np.asarray(a[2],dtype=np.uint8)])
            pca.extend([np.asarray(a[3],dtype=np.uint8),
                        np.flipud(np.fliplr(a[1])),
                        np.flipud(np.fliplr(a[0])),
                        np.asarray(a[2],dtype=np.uint8)])
        d['user'] = {'pixel_map':np.asarray(np.pad(pca,((0,0),(0,2),(0,0))),dtype=np.uint8).ravel().tolist()}
    else:
        d['user'] = {'pixel_map':e.ravel().tolist()}
    print('pixel_map {}'.format(len(d['user']['pixel_map'])))
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update gain map')
    parser.add_argument('--file' , help='input pixel mask', type=str, required=False, default=None)
    parser.add_argument('--gain', help='gain for pixels [0 1]', default=['M','L'], nargs=2, choices=gain_dict.keys(), required=True)
    parser.add_argument('--dev', help='use development db', action='store_true')
    parser.add_argument('--inst', help='instrument', type=str, default='rix')
    parser.add_argument('--alias', help='alias name (BEAM)', type=str, default='BEAM')
    parser.add_argument('--name', help='detector name (epixhr)', type=str, default='epixhr')
    parser.add_argument('--segm', help='detector segment (0)', type=int, default=0)
    parser.add_argument('--id', help='device id/serial num', type=str, default='serial1234')
    parser.add_argument('--user', help='user for HTTP authentication', type=str, default='rixopr')
    parser.add_argument('--password', help='password for HTTP authentication', type=str, default=os.getenv('CONFIGDB_AUTH'))
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

    #  Write gainmap
    if args.file is None:
        d = {}
        trbit = gain_dict[args.gain[0]]['trbit']
        d['expert'] = {'EpixHR':{}}
        for i in range(4):
            d['expert']['EpixHR'][f'Hr10kTAsic{i}'] = {'trbit':trbit}

        pixelMap = np.zeros((elemRows*2,elemCols*2),dtype=np.uint8)+gain_dict[args.gain[0]]['value']
        d['user'] = {'pixel_map':pixelMap.ravel().tolist()}
    else:
        d = epixhr_readmap(args.file,args.gain)
    print('copyValues from d')
    copyValues(d['user'],cfg,'user')
    copyValues(d['expert'],cfg,'expert')

    ntrbit = [ cfg['expert']['EpixHR'][f'Hr10kTAsic{i}']['trbit'] for i in range(4)]

    print(f'Changed trbit to {ntrbit}')

    print('--New pixel map--')
    print(cfg['user']['pixel_map'])

    if not args.test:
        mycdb.modify_device(args.alias, cfg)

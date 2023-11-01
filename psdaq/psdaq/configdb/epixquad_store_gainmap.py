import os

from psdaq.configdb.epixquad_cdict import epixquad_cdict
import psdaq.configdb.configdb as cdb
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

gain_dict = {'H'  :{'value':0xc,'trbit':1},
             'M'  :{'value':0xc,'trbit':0},
             'L'  :{'value':0x8,'trbit':0},
             'AHL':{'value':0  ,'trbit':1},
             'AML':{'value':0  ,'trbit':0},}

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

def epixquad_readmap(fname,gains):
    d = {}
    gain_to_value = (0xc,0xc,0x8,0x0,0x0) # H/M/L/AHL/AML
    gain_to_trbit = (0x1,0x0,0x0,0x1,0x0) # H/M/L/AHL/AML

    if gain_dict[gains[0]]['trbit']!=gain_dict[gains[1]]['trbit']:
        raise ValueError(f'Incompatible gains {gains} for pixel configuration')

    trbit = gain_dict[gains[0]]['trbit']
    for i in range(16):
        d[f'expert.EpixQuad.Epix10kaSaci{i}.trbit'] = trbit

    _height = 192
    _width  = 176

    vgain0 = gain_dict[gains[0]]['value']
    vgain1 = gain_dict[gains[1]]['value']

    e = np.genfromtxt(fname,dtype=np.uint8)
    e = e*vgain1 + (vgain0-vgain1)
    e = np.vsplit(e,4)
    
    #  break the elements into asics
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

    d['user.pixel_map'] = np.asarray(np.pad(pca,((0,0),(0,2),(0,0))),dtype=np.uint8)
    return d

def main():
    parser = argparse.ArgumentParser(description='Update gain map')
    parser.add_argument('--file' , help='input pixel mask', type=str, required=True)
    parser.add_argument('--gain', help='gain for pixels [0 1]', default=['M','L'], nargs=2, choices=gain_dict.keys(), required=True)
    parser.add_argument('--dev', help='use development db', action='store_true')
    parser.add_argument('--inst', help='instrument', type=str, default='ued')
    parser.add_argument('--alias', help='alias name', type=str, default='BEAM')
    parser.add_argument('--name', help='detector name', type=str, default='epixquad')
    parser.add_argument('--segm', help='detector segment', type=int, default=0)
    parser.add_argument('--id', help='device id/serial num', type=str, default='serial1234')
    parser.add_argument('--user', help='user for HTTP authentication', type=str, default='uedopr')
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

    top = epixquad_cdict()  # We need the full cdict in order to store
    copyValues(cfg,top)     # Now copy old values into the cdict

    #  Write gainmap
    d = epixquad_readmap(args.file,args.gain)
    copyValues(d,top)

    print('Setting user.pixel_map')
    top.set('user.pixel_map', d['user.pixel_map'])

    #  Store
    top.setInfo('epix10kaquad', args.name, args.segm, args.id, 'No comment')
    if not args.test:
        mycdb.modify_device(args.alias, top)

if __name__ == "__main__":
    main()

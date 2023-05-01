import psdaq.configdb.configdb as cdb
import sys
import IPython
import numpy as np
import argparse

#  Copy values and shape from config dict into cdict
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

def main(name, cdict_fn):
    parser = argparse.ArgumentParser(description='Update weights and/or calib polynomial constants')
    parser.add_argument('--weights', help='space-delimited file of weights', default='')
    parser.add_argument('--calib', help='space-delimited file of coefficients', default='')

    parser.add_argument('--dev', help='use development db', action='store_true')
    parser.add_argument('--inst', help='instrument', type=str, default='tmo')
    parser.add_argument('--alias', help='alias name', type=str, default='BEAM')
    parser.add_argument('--name', help='detector name', type=str, default='tmoopal2')
    parser.add_argument('--segm', help='detector segment', type=int, default=0)
    parser.add_argument('--id', help='device id/serial num', type=str, default='serial1234')
    parser.add_argument('--user', help='user for HTTP authentication', type=str, default='tstopr')
    parser.add_argument('--password', help='password for HTTP authentication', type=str, default='pcds')

    args = parser.parse_args()

    weights = np.loadtxt(args.weights)
    calib   = np.loadtxt(args.calib)

    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.
    detname = f'{args.name}_{args.segm}'

    db   = 'devconfigdb' if args.dev else 'configdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    create = False

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    cfg = mycdb.get_configuration(args.alias,detname)

    if cfg is None: raise ValueError('Config for instrument/detname %s/%s not found. dbase url: %s, db_name: %s, config_style: %s'%(args.inst,detname,url,dbname,args.alias))

    top = cdict_fn()

    #  Need our own function to copy into top
    copyValues(cfg,top)

    if len(weights.shape)==1:
        if weights.shape[0]>0:
            print(f'Storing weights of length {weights.shape[0]}')
            top.set('fex.fir_weights', weights, 'DOUBLE', override=True)
        else:
            print('Weights not updated')
    else:
        raise ValueError('dimension of weights {} is > 1'.format(len(weights.shape)))


    if len(calib.shape)==1:
        if calib.shape[0]>0:
            print(f'Storing calib of length {calib.shape[0]}')
            top.set('fex.calib_poly', calib, 'DOUBLE', override=True)
        else:
            print('Calib not updated')
    else:
        raise ValueError('dimension of calib {} is > 1'.format(len(calib.shape)))

    top.setInfo(name, args.name, args.segm, args.id, 'No comment')
    mycdb.modify_device(args.alias, top)


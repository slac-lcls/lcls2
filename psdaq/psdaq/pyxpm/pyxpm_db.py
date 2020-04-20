from psdaq.configdb.typed_json import cdict
from psdaq.configdb.get_config import get_config_with_params
import psdaq.configdb.configdb as cdb
import os
import io
import argparse

create = True

parser = argparse.ArgumentParser()
parser.add_argument('--url',type=str,default='https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/',help='DB URL')
parser.add_argument('--inst',type=str,default='Lab2',help='instrument')
parser.add_argument('--alias',type=str,default='PROD',help='alias')
parser.add_argument('--db',type=str,default='configDB',help='database')
parser.add_argument('--load',action='store_true',help='load from db')
parser.add_argument('--store',action='store_true',help='store to db')
parser.add_argument('--name',type=str,default='DAQ:LAB2:XPM:0',help='device name')
args = parser.parse_args()

if args.store:

    mycdb = cdb.configdb(args.url,args.inst,create,args.db)
    mycdb.add_device_config('xpm')

    top = cdict()

    top.setInfo('xpm', args.name, detSegm=None, 'serial1234', 'No comment')
    top.setAlg('config', [0,0,0])

    top.set('XTPG.CuDelay'   , 160000, 'UINT32')
    top.set('XTPG.CuBeamCode',     40, 'UINT8')
    top.set('XTPG.CuInput'   ,      2, 'UINT8')
    v = [90]*8
    top.set('PART.L0Delay', v, 'UINT32')

    if not args.alias in mycdb.get_aliases():
        mycdb.add_alias(args.alias)
    mycdb.modify_device(args.alias, top)

    mycdb.print_configs()

if args.load:

    print('Loading with url {:}  inst {:}  db {:}  alias {:}  name {:}'.format(args.url, args.inst, args.db, args.alias, args.name))
    cfg = get_config_with_params(args.url, args.inst, args.db, args.alias, args.name)
    print('Loaded {:}'.format(cfg))

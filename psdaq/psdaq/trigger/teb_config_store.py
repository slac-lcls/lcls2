from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import os
import io
import argparse

parser = argparse.ArgumentParser(description='Write a new teb configuration into the database')
parser.add_argument('--inst', help='instrument', type=str, default='tst')
parser.add_argument('--alias', help='alias name', type=str, default='BEAM')
parser.add_argument('--name', help='detector name', type=str, default='tmoTeb')
parser.add_argument('--segm', help='detector segment', type=int, default=0)
parser.add_argument('--id', help='device id/serial num', type=str, default='No serial number')
args = parser.parse_args()

# these are the current default values, but put them here to be explicit
create = False
dbname = 'configDB'

mycdb = cdb.configdb('https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/', args.inst, create, dbname)

# this needs to be called once per detType at the
# "beginning of time" to create the collection name (same as detType
# in top.setInfo).  It doesn't hurt to call it again if the collection
# already exists, however.
mycdb.add_device_config('teb')

top = cdict()

top.setInfo('teb', args.name, args.segm, args.id, 'No comment')
top.setAlg('tebConfig', [0,0,1])

top.set('soname', 'libtmoTeb.so', 'CHARSTR')

# This is a required entry:
top.set('prescale', 1, 'UINT32')

top.set('persistValue', 0xdeadbeef, 'UINT32')
top.set('monitorValue', 0x12345678, 'UINT32')

mycdb.add_alias(args.alias)
mycdb.modify_device(args.alias, top)
mycdb.print_configs()

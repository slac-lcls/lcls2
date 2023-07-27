# Load with:
#  python teb_config_store.py --prod --inst tst --name trigger --segm 0 --alias BEAM --user tstopr --password pcds

from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import os
import io

# these are the current default values, but put them here to be explicit
create = False
dbname = 'configDB'

args = cdb.createArgs().args
db   = 'configdb' if args.prod else 'devconfigdb'
url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

#mycdb = cdb.configdb('https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/', args.inst, create,
#                     root=dbname, user=args.user, password=args.password)
mycdb = cdb.configdb(url, args.inst, create,
                     root=dbname, user=args.user, password=args.password)
mycdb.add_alias(args.alias)

# this needs to be called once per detType at the
# "beginning of time" to create the collection name (same as detType
# in top.setInfo).  It doesn't hurt to call it again if the collection
# already exists, however.
mycdb.add_device_config('teb')

top = cdict()

top.setInfo('teb', args.name, args.segm, args.id, 'No comment')
top.setAlg('tebConfig', [0,1,0])

help_str  = "-- user --"
help_str += "\nsoname       : The trigger library the TEB is to use"
help_str += "\n               It is to be found in $TESTRELDIR"
help_str += "\npythonScript : Trigger script to run given a suitable trigger library"
help_str += "\n               Its path is given by the TEB's 'script_path' kwarg"
help_str += "\nbuildAll     : Event build all detector contributions vs only"
help_str += "\n               those listed in 'buildDets'"
help_str += "\nbuildDets    : Comma separated list of detNames to event build"
help_str += "\n               Ignored when buildAll is 0"
help_str += "\nprescale     : Record 1 in N events for which the persist trigger"
help_str += "\n               condition isn't met"
help_str += "\npersistValue : Trigger condition for recording an event"
help_str += "\nmonitorValue : Trigger condition for monitoring an event"
help_str += "\nrogRsrvdBuf  : Number of MEB event buffers to be held aside for"
help_str += "\n               events to which a slow readout group contributed"
top.set('help:RO', help_str, 'CHARSTR')

top.set('soname', 'libtmoTeb.so', 'CHARSTR')

top.set('pythonScript', 'tebTstTrigger.py', 'CHARSTR')

top.set('buildAll', 1, 'UINT32')
top.set('buildDets', 'timing,bld,epics', 'CHARSTR')

top.set('prescale', 1, 'UINT32')

top.set('persistValue', 0xdeadbeef, 'UINT32')
top.set('monitorValue', 0x12345678, 'UINT32')

for rog in range(8):
    top.set(f'rogRsrvdBuf[{str(rog)}]', 0, 'UINT32')

mycdb.add_alias(args.alias)
mycdb.modify_device(args.alias, top)
#mycdb.print_configs()

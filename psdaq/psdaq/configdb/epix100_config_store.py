from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import numpy as np
import sys
import IPython
import argparse

def epix100_cdict():

    top = cdict()
    top.setAlg('config', [2,0,0])

    #top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    #top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- user interface --"
    help_str += "\nstart_ns     : nanoseconds to exposure start"

    top.set("user.start_ns" , 107749, 'UINT32')

    # timing system
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.PauseThreshold',16,'UINT32')
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.TriggerDelay',42,'UINT32')
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.Partition',0,'UINT32')

    return top

if __name__ == "__main__":
    create = True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args

    db = 'configdb' if args.prod else 'devconfigdb'
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('epix100')

    top = epix100_cdict()
    top.setInfo('epix100', args.name, args.segm, args.id, 'No comment')

    mycdb.modify_device(args.alias, top)

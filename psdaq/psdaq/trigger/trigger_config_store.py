# This is an example configuration meant for use with the libtmoTrigger.so library
# It is not normally used and may be out of date with respect to the DRP & TEB code
#
# Load with:
#  python trigger_config_store.py --inst tst --name trigger --segm 0 --alias BEAM --user tstopr

from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import os
import io

def usual_cdict():
    top = cdict()

    top.setAlg('triggerConfig', [0,0,1])

    help_str  = "-- user --"
    help_str += "\nsoname       : The trigger library the DRPs and TEBs are to use"
    help_str += "\n               It is to be found in $TESTRELDIR"
    help_str += "\nprescale     : Record 1 in N events for which the persist trigger"
    help_str += "\n               condition isn't met"
    help_str += "\npersistValue : Value DRP must provide to have TEB issue a"
    help_str += "\n               persist trigger"
    help_str += "\nmonitorValue : Value DRP must provide to have TEB issue a"
    help_str += "\n               monitor trigger"
    help_str += "\npeaksThresh  : Threshold HSD DRPs must exceed to have TEB"
    help_str += "\n               issue a persist trigger"
    help_str += "\nebeamThresh  : Threshold BLD DRP must exceed to have TEB"
    help_str += "\n               issue a persist trigger"
    help_str += "\ncam          : Integer 'name' to assign to DRPs with detName 'cam'"
    help_str += "\nxpphsd       : Integer 'name' to assign to DRPs with detName 'xpphsd'"
    help_str += "\nbld          : Integer 'name' to assign to DRPs with detName 'bld'"
    top.set('help:RO', help_str, 'CHARSTR')

    top.set('soname', 'libtmoTrigger.so',  'CHARSTR')

    # This is a required entry by the TEB:
    top.set('prescale', 1000, 'UINT32')

    # Detector names handled by this library must be listed here
    #  Values are arbitrary but must match the trigger code's usage
    #  This is how a detName is recognized from its source ID
    top.set('cam',    0, 'UINT32')
    top.set('xpphsd', 1, 'UINT32')
    top.set('bld',    2, 'UINT32')

    # CAM trigger parameters:
    top.set('persistValue', 0xdeadbeef, 'UINT32')
    top.set('monitorValue', 0x12345678, 'UINT32')

    # HSD trigger parameters:
    top.set('peaksThresh', 3, 'UINT32')

    # BLD trigger parameters:
    top.set('eBeamThresh', 7, 'UINT32')

    return top

def calib_cdict():
    top = cdict()

    top.setAlg('triggerConfig', [0,0,0])

    help_str  = "-- expert --"
    help_str += "\nsoname       : The trigger library the DRPs and TEBs are to use"
    help_str += "\n               It is to be found in $TESTRELDIR"
    top.set('help:RO', help_str, 'CHARSTR')

    top.set('soname', 'libcalibTrigger.so',  'CHARSTR')

    top.set('buildAll:RO', 1, 'UINT32') # Required parameter
    top.set('prescale:RO', 1, 'UINT32') # Required parameter

    return top

if __name__ == "__main__":
    # these are the current default values, but put them here to be explicit
    create = False          # Set to True only for the first time
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args
    db   = 'configdb' if args.prod else 'devconfigdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)

    # this needs to be called once per detType at the
    # "beginning of time" to create the collection name (same as detType
    # in top.setInfo).  It doesn't hurt to call it again if the collection
    # already exists, however.
    mycdb.add_device_config('trigger')

    if args.alias == 'CALIB':
        top = calib_cdict()
    else:
        top = usual_cdict()
    top.setInfo('trigger', args.name, args.segm, args.id, 'No comment')

    mycdb.modify_device(args.alias, top)
    mycdb.print_configs()

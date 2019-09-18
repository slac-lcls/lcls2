from psalg.configdb.typed_json import cdict
import psalg.configdb.configdb as cdb
import os
import io

# these are the current default values, but put them here to be explicit
create = False
dbname = 'configDB'
instrument = 'TMO'

mycdb = cdb.configdb('mcbrowne:psana@psdb-dev:9306', instrument, create, dbname)

# this needs to be called once per detType at the
# "beginning of time" to create the collection name (same as detType
# in top.setInfo).  It doesn't hurt to call it again if the collection
# already exists, however.
mycdb.add_device_config('trigger')

top = cdict()

top.setInfo('trigger', 'trigger', 'No serial number', 'No comment')
top.setAlg('triggerConfig', [0,0,1])

top.set('soname', 'libtmoTrigger.so',  'CHARSTR')

# This is a required entry:
top.set('prescale', 1000, 'UINT32')

# Detector names handled by this library must be listed here
#  Values are arbitrary but must match the trigger code's usage
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

mycdb.modify_device('BEAM', top)
mycdb.print_configs()

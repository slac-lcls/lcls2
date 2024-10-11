from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import os
import io

create = True
dbname = 'configDB'
args = cdb.createArgs().args
db   = 'configdb' if args.prod else 'devconfigdb'
url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

top = cdict()

top.setAlg('config', [3,0,0])

top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
top.set("firmwareVersion:RO",   0, 'UINT32' )

help_str = "-- user fields --"
help_str += "\nuser.raw.start_ns : nanoseconds from fiducial to sampling start"
help_str += "\nuser.raw.gate_ns  : nanoseconds from sampling start to end"
help_str += "\nuser.raw.prescale : event downsampling; record 1-out-of-N"
help_str += "\nuser.raw.keep     : event downsampling; record on keepraw event"
help_str += "\nuser.fex.start_ns : nanoseconds from fiducial to sparsify start"
help_str += "\nuser.fex.gate_ns  : nanoseconds from sparsify start to end"
help_str += "\nuser.fex.prescale : event downsampling; record 1-out-of-N"
help_str += "\nuser.fex.corr.baseline : new baseline after sample correction"
help_str += "\nuser.fex.corr.accum    : pre-gate baseline accumulation period"
help_str += "\nuser.fex.ymin     : minimum ADC value to sparsify"
help_str += "\nuser.fex.ymax     : maximum ADC value to sparsify"
help_str += "\nuser.fex.xpre     : keep N samples leading excursion"
help_str += "\nuser.fex.xpost    : keep N samples trailing excursion"

top.set("help:RO", help_str, 'CHARSTR')

top.set('user.raw.start_ns',  93000, 'UINT32')
top.set('user.raw.gate_ns' ,    200, 'UINT32')
top.set('user.raw.prescale',      1, 'UINT32')
top.set('user.raw.keep',          1, 'UINT32')

top.set('user.fex.start_ns',  93000, 'UINT32')
top.set('user.fex.gate_ns' ,    200, 'UINT32')
top.set('user.fex.prescale',      0, 'UINT32')
top.set('user.fex.ymin' ,      2000, 'UINT32')
top.set('user.fex.ymax' ,      2080, 'UINT32')
top.set('user.fex.xpre' ,         8, 'UINT32')
top.set('user.fex.xpost',         8, 'UINT32')

top.define_enum('baselineAccumEnum', {'  11 ns ( 64 S)' :  4, 
                                      '  22 ns (128 S)' :  5, 
                                      '  43 ns (256 S)' :  6, 
                                      '  86 ns (512 S)' :  7,
                                      ' 172 ns (  1 kS)':  8, 
                                      ' 345 ns (  2 kS)':  9, 
                                      ' 689 ns (  4 kS)': 10, 
                                      '1378 ns (  8 kS)': 11, 
                                      '2757 ns ( 16 kS)': 12})

top.set('user.fex.corr.baseline', 16384, 'UINT32')
top.set('user.fex.corr.accum',       12, 'baselineAccumEnum')

top.define_enum('dataModeEnum', {'Data': -1, 'Ramp': 0, 'Spike11': 1, 'Spike12': 3, 'Spike16': 5})

top.set('expert.readoutGroup', 0, 'UINT32')
top.set('expert.enable'      , 0, 'UINT32')
top.set('expert.raw_start'   , 40, 'UINT32')
top.set('expert.raw_gate'    , 40, 'UINT32')
top.set('expert.raw_prescale', 0, 'UINT32')
top.set('expert.raw_keep'    , 0, 'UINT32')
top.set('expert.fex_start'   , 40, 'UINT32')
top.set('expert.fex_gate'    , 40, 'UINT32')
top.set('expert.fex_xpre'    , 1, 'UINT32')
top.set('expert.fex_xpost'   , 1, 'UINT32')
top.set('expert.fex_ymin'    , 2020, 'UINT32')
top.set('expert.fex_ymax'    , 2060, 'UINT32')
top.set('expert.fex_prescale', 0, 'UINT32')
top.set('expert.test_pattern', -1, 'dataModeEnum')
top.set('expert.full_event'  , 6, 'UINT32')
top.set('expert.full_size'   , 3072, 'UINT32')
top.set('expert.fs_range_vpp', 65535, 'UINT32')
top.set('expert.trig_shift'  , 0, 'UINT32')
top.set('expert.sync_ph_even', 0, 'UINT32')
top.set('expert.sync_ph_odd' , 0, 'UINT32')

if False:
    top.set('adccal.oadj_a_vina', 0x0800, 'UINT16')
    top.set('adccal.oadj_a_vinb', 0x0800, 'UINT16')
    top.set('adccal.oadj_b_vina', 0x0800, 'UINT16')
    top.set('adccal.oadj_b_vinb', 0x0800, 'UINT16')
    top.set('adccal.gain_trim_a', 0x80, 'UINT8')
    top.set('adccal.gain_trim_b', 0x80, 'UINT8')
    top.set('adccal.b0_time_0'  , 0x80, 'UINT8')
    top.set('adccal.b0_time_90' , 0x80, 'UINT8')
    top.set('adccal.b1_time_0'  , 0x80, 'UINT8')
    top.set('adccal.b1_time_90' , 0x80, 'UINT8')
    top.set('adccal.b4_time_0'  , 0x80, 'UINT8')
    top.set('adccal.b4_time_90' , 0x80, 'UINT8')
    top.set('adccal.b5_time_0'  , 0x80, 'UINT8')
    top.set('adccal.b5_time_90' , 0x80, 'UINT8')
    top.set('adccal.tadj_a_fg90', 0x80, 'UINT8')
    top.set('adccal.tadj_b_fg0' , 0x80, 'UINT8')

top.setInfo('hsd', args.name, args.segm, args.id, 'No comment')

if args.update:
    cfg = mycdb.get_configuration(args.alias, args.name+'_%d'%args.segm)
    top = update_config(cfg, top.typed_json(), args.verbose)

if not args.dryrun:
    if create:
        mycdb.add_alias(args.alias)
        mycdb.add_device_config('hsd')
    mycdb.modify_device(args.alias, top)



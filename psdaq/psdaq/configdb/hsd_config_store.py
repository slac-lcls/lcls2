from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
from psdaq.configdb.get_config import update_config
import os
import io

create = True
dbname = 'configDB'
args = cdb.createArgs().args
db   = 'configdb' if args.prod else 'devconfigdb'
url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

mycdb = cdb.configdb(url, args.inst, create,
                     root=dbname, user=args.user, password=args.password)

top = cdict()

top.setAlg('config', [3,3,0])

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
help_str += "\nuser.fex.dymin    : min ADC value to sparsify relative to fex.corr.baseline"
help_str += "\nuser.fex.dymax    : max ADC value to sparsify relative to fex.corr.baseline"
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
top.set('user.fex.dymin' ,      -20, 'INT32')
top.set('user.fex.dymax' ,       20, 'INT32')
top.set('user.fex.xpre' ,         8, 'UINT32')
top.set('user.fex.xpost',         8, 'UINT32')

top.define_enum('accumEnum', {'11ns_64Sa'   :  4, 
                              '22ns_128Sa'  :  5, 
                              '43ns_256Sa'  :  6, 
                              '86ns_512Sa'  :  7,
                              '172ns_1kSa'  :  8, 
                              '345ns_2kSa'  :  9, 
                              '689ns_4kSa'  : 10, 
                              '1378ns_8kSa' : 11, 
                              '2757ns_16kSa': 12})

top.set('user.fex.corr.baseline' , 16384, 'UINT32')
top.set('user.fex.corr.accum'    ,    12, 'accumEnum')

top.define_enum('dataModeEnum', {'Data': -1, 'Ramp': 0, 'Spike11': 1, 'Spike12': 3, 'Spike16': 5})

top.set('expert.readoutGroup' , 0    , 'UINT32')
top.set('expert.enable'       , 0    , 'UINT32')
top.set('expert.raw_start'    , 40   , 'UINT32')
top.set('expert.raw_gate'     , 40   , 'UINT32')
top.set('expert.raw_prescale' , 0    , 'UINT32')
top.set('expert.raw_keep'     , 0    , 'UINT32')
top.set('expert.fex_start'    , 40   , 'UINT32')
top.set('expert.fex_gate'     , 40   , 'UINT32')
top.set('expert.fex_xpre'     , 1    , 'UINT32')
top.set('expert.fex_xpost'    , 1    , 'UINT32')
top.set('expert.fex_ymin'     , 2020 , 'UINT32')
top.set('expert.fex_ymax'     , 2060 , 'UINT32')
top.set('expert.fex_prescale' , 0    , 'UINT32')
top.set('expert.test_pattern' , -1   , 'dataModeEnum')
top.set('expert.full_rtt'     , 300  , 'UINT32')  # DT round trip time @186MHz
top.set('expert.full_event'   , 6    , 'UINT32')
top.set('expert.full_size_raw', 3072 , 'UINT32')
top.set('expert.full_size_fex', 3072 , 'UINT32')
top.set('expert.fs_range_vpp' , 65535, 'UINT32')

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
    ofex = cfg['user']['fex']
    nfex = top['user']['fex']
    bline = int(ofex['corr']['baseline'])
    if 'dymin' not in ofex:
        nfex['dymin'] = ofex['ymin']-bline
        nfex['dymax'] = ofex['ymax']-bline

if not args.dryrun:
    if create:
        mycdb.add_alias(args.alias)
        mycdb.add_device_config('hsd')
    mycdb.modify_device(args.alias, top)

if args.verbose:
    print('--- new ---')
    import pprint
    pprint.pp(top)


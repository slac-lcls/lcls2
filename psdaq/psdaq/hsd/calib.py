from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
from psdaq.configdb.get_config import get_config_with_params
import time
import json

def get_calib(db_url,instrument,seg,alias):

    result = ''
    try:
        cal = get_config_with_params(db_url, instrument, 'configDB', alias, seg)
        result = json.dumps(cal)
    except:
        print('Failed to load calib constants {:}'.format([db_url,instrument,seg,alias]))
        pass
    return result

def set_calib(db_url,instrument,seg,alias,calib):

    dict = json.loads(calib)

    create = True
    dbname = 'configDB'

    mycdb = cdb.configdb(db_url, instrument, create,
                         root='configDB', user=instrument+'opr', password='pcds')
    mycdb.add_device_config('hsdcal')

    top = cdict()

    top.setAlg('calib', [0,0,0])

    help_str = "No help at this time"
    top.set("help:RO", help_str, 'CHARSTR')

    for k,v in dict['expert'].items():
        top.set("expert."+k, v, 'UINT16')

    mycdb.add_alias(alias)

    name = seg.split('_')[0]
    segn = int(seg.split('_')[1])
    top.setInfo('hsd', name, segn, 'serial1234', 'No comment')
    mycdb.modify_device(alias, top)

if __name__=='__main__':

    args = cdb.createArgs().args
    db   = 'configdb' if args.prod else 'devconfigdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'
    
    calib = {}
    calib['OADJ_A_VINA'] = 0x07dd
    calib['OADJ_A_VINB'] = 0x0793
    calib['OADJ_B_VINA'] = 0x081f
    calib['OADJ_B_VINB'] = 0x07e0
    calib['GAIN_TRIM_A'] = 0x93
    calib['GAIN_TRIM_B'] = 0x8d
    calib['B0_TIME_0'  ] = 0x6c
    calib['B0_TIME_90' ] = 0x91
    calib['B1_TIME_0'  ] = 0x80
    calib['B1_TIME_90' ] = 0x80
    calib['B4_TIME_0'  ] = 0x94
    calib['B4_TIME_90' ] = 0x80
    calib['B5_TIME_0'  ] = 0x80
    calib['B5_TIME_90' ] = 0x80
    calib['TADJ_A_FG90'] = 0xc3
    calib['TADJ_B_FG0' ] = 0x80

    dict = {}
    dict['expert'] = calib
    scalib = json.dumps(dict)
    print('calib {:}'.format(scalib))
    set_calib(url, args.inst, args.name+'_%d'%args.segm, 'CALIB', scalib)

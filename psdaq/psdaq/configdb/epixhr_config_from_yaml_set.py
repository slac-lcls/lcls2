from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import pyrogue as pr
import numpy as np
import sys
import IPython
import argparse
from epixhr_config_from_yaml import copyValues

path = '/cds/home/w/weaver/epixhr/epix-hr-single-10k/yml/'
files = {}
files['320MHz'] = [path+'ePixHr10kT_MMCM_320MHz.yml',
                   path+'ePixHr10kT_PowerSupply_Enable.yml',
                   path+'ePixHr10kT_RegisterControl_24us_320MHz.yml',
                   path+'ePixHr10kT_PacketRegisters.yml',
                   path+'ePixHr10kT_PLLBypass_320MHz_ASIC_0.yml',
                   path+'ePixHr10kT_PLLBypass_320MHz_ASIC_1.yml',
                   path+'ePixHr10kT_PLLBypass_320MHz_ASIC_2.yml',
                   path+'ePixHr10kT_PLLBypass_320MHz_ASIC_3.yml',
#                   path+'ePixHr10kT_TriggerRegisters_100Hz.yml',
]

files['160MHz'] = [path+'ePixHr10kT_MMCM_160MHz.yml',
                   path+'ePixHr10kT_PowerSupply_Enable.yml',
                   path+'ePixHr10kT_RegisterControl_24us_160MHz.yml',
                   path+'ePixHr10kT_PacketRegisters.yml',
                   path+'ePixHr10kT_PLLBypass_160MHz_ASIC_0.yml',
                   path+'ePixHr10kT_PLLBypass_160MHz_ASIC_1.yml',
                   path+'ePixHr10kT_PLLBypass_160MHz_ASIC_2.yml',
                   path+'ePixHr10kT_PLLBypass_160MHz_ASIC_3.yml',
#                   path+'ePixHr10kT_TriggerRegisters_100Hz.yml',
]
fload = files['320MHz']

create = False
dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

args = cdb.createArgs().args
args.inst = 'asc'
args.prod = False
args.name = 'epixhr'
args.segm = 0
args.user = 'detopr'
args.password = 'pcds'

db = 'configdb' if args.prod else 'devconfigdb'
mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,
                     root=dbname, user=args.user, password=args.password)
top = mycdb.get_configuration(args.alias, args.name+'_%d'%args.segm)

base = 'ePixHr10kT'
for fn in fload:
    d = pr.yamlToData(fName=fn)
    copyValues(d[base],top,'expert')

mycdb.modify_device(args.alias, top)

    

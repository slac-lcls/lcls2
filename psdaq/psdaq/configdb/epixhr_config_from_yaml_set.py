from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import pyrogue as pr
import numpy as np
import sys
import IPython
import argparse

#
#  Edit here
#
#  'path' is the directory where to find the yaml files
#
#path = '/cds/home/w/weaver/epixhr/epix-hr-single-10k/yml/'
path = '/cds/home/w/weaver/epix-hr-new/software/yml/'

#
#  Put a dictionary here to hold several different configuration sets
#
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

files['248MHz'] = [path+'ePixHr10kT_MMCM_248MHz.yml',
                   path+'ePixHr10kT_PowerSupply_Enable.yml',
                   path+'ePixHr10kT_RegisterControl_24us_248MHz.yml',
                   path+'ePixHr10kT_PacketRegisters.yml',
                   path+'ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml',
                   path+'ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml',
                   path+'ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml',
                   path+'ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml',
#                   path+'ePixHr10kT_TriggerRegisters_100Hz.yml',
]

files['4'] = [path+'ePixHr10kT_MMCM_248MHz.yml',
              path+'ePixHr10kT_PowerSupply_Enable.yml',
              path+'ePixHr10kT_RegisterControl_24us_248MHz.yml',
              path+'ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml',
              path+'ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml',
              path+'ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml',
              path+'ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml',
              path+'ePixHr10kT_SSP.yml',
              path+'ePixHr10kT_PacketRegisters.yml',
              #                   path+'ePixHr10kT_TriggerRegisters_100Hz.yml',
]

#
#  Choose which configuration set with 'fload'
#
fload = files['4']

#
#  No more editing below
#

def copyValues(din,dout,k=None):
    if isinstance(din,dict) and isinstance(dout[k],dict):
        for key,value in din.items():
            if key in dout[k]:
                copyValues(value,dout[k],key)
            else:
                print(f'skip {key}')
    elif isinstance(din,bool):
        vin = 1 if din else 0
        if dout[k] != vin:
            print(f'Writing {k} = {vin}')
            dout[k] = 1 if din else 0
        else:
            print(f'{k} unchanged')
    else:
        if dout[k] != din:
            print(f'Writing {k} = {din}')
            dout[k] = din
        else:
            print(f'{k} unchanged')

create = False
dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

args = cdb.createArgs().args
#args.user = 'rixopr'
#args.inst = 'rix'
#args.prod = True
args.user = 'tstopr'
args.inst = 'tst'
args.prod = True
args.name = 'epixhr'
args.segm = 0
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

    

from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import pyrogue as pr
import numpy as np
from itertools import chain
import sys
import IPython
import argparse

#
#  Edit here
#
#  'path' is the directory where to find the yaml files
#
path = '/cds/home/w/weaver/epix-hr-new/software'

#
#  Put a dictionary here to hold several different configuration sets
#
class ePixYml(object):
    def __init__(self,arg):
        # copy and paste from epix-hr-single10k/software/python/ePixFpga
        arguments = np.asarray(arg)
        if arguments[0] == 1:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_320MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_24us_320MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_320MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_320MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_320MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_320MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 2:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_320MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_150us_320MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 3:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_320MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_24us_320MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 4:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_320MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_24us_248MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 5:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_320MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width12us_AcqWidth24us_320MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 6:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_320MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width12us_AcqWidth24us_5p18kHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"                    
        if arguments[0] == 11:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_307MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width24us_AcqWidth24us_4p87kHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 12:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_307MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width12us_AcqWidth24us_5p18kHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"           
        if arguments[0] == 13:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_301MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width24us_AcqWidth24us_4p87kHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 14:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_301MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width12us_AcqWidth24us_5p18kHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 15:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_280MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_24us_248MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 16:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_271MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_24us_248MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 17:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_262MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width24us_AcqWidth24us_4p00kHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 18:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_262MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width12us_AcqWidth24us_4p46kHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 21:
            self.filenameMMCM              = "yml/ePixHr10kT_MMCM_248MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_24us_248MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 22:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_248MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_150us_248MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 23:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_248MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_R0Width12us_AcqWidth24us_248MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_248MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"
        if arguments[0] == 31:
            self.filenameMMCM              = "/yml/ePixHr10kT_MMCM_160MHz.yml"
            self.filenamePowerSupply       = "/yml/ePixHr10kT_PowerSupply_Enable.yml"
            self.filenameRegisterControl   = "/yml/ePixHr10kT_RegisterControl_24us_160MHz.yml"
            self.filenameASIC0             = "/yml/ePixHr10kT_PLLBypass_160MHz_ASIC_0.yml"
            self.filenameASIC1             = "/yml/ePixHr10kT_PLLBypass_160MHz_ASIC_1.yml"
            self.filenameASIC2             = "/yml/ePixHr10kT_PLLBypass_160MHz_ASIC_2.yml"
            self.filenameASIC3             = "/yml/ePixHr10kT_PLLBypass_160MHz_ASIC_3.yml"
            self.filenameSSP               = "/yml/ePixHr10kT_SSP.yml"
            self.filenamePacketReg         = "/yml/ePixHr10kT_PacketRegisters.yml"
            self.filenameTriggerReg        = "/yml/ePixHr10kT_TriggerRegisters_100Hz.yml"

        self.files = [self.filenameMMCM,
                      self.filenamePowerSupply,
                      self.filenameRegisterControl,
                      self.filenameASIC0,
                      self.filenameASIC1,
                      self.filenameASIC2,
                      self.filenameASIC3,
                      self.filenameSSP,
                      self.filenamePacketReg,
#                      self.filenameTriggerReg,
        ]

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

args = cdb.createArgs(inst='rix',prod=True,name='epixhr',segm=0,user='rixopr',yaml='5').args

db = 'configdb' if args.prod else 'devconfigdb'
mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,
                     root=dbname, user=args.user, password=args.password)
top = mycdb.get_configuration(args.alias, args.name+'_%d'%args.segm)

base = 'ePixHr10kT'

yml = ePixYml([int(args.yaml)])
for f in yml.files:
    print(path+f)

for fn in yml.files:
    d = pr.yamlToData(fName=path+fn)
    copyValues(d[base],top,'expert')

mycdb.modify_device(args.alias, top)

    

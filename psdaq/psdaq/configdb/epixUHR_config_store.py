'''python epixUHR_config_store.py --user tstopr --inst tst --alias BEAM --name epixuhr --segm 0 --id epixuhr_serial1234'''


from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import numpy as np
import sys
import IPython
import argparse
import functools
from os import walk
import yaml

elemRows = 168
elemCols = 192

ymlfilename={}
ymlfilename['_35kHz']={
            'RegisterControl'   : "config/UhrWaveformControlAxi_registers.yml",
            'TriggerReg'        : "config/UhrTrigControlAxi_registers.yml",
            'SACIReg'           : "config/ePixUHR_SACI_Registers.yml",
            'FramerReg'         : "config/UhrFramerAxi_registers.yml",
            'General'           : "config/ePixUHR_camera_general_settings.yml",
}
        
ymlfilename['_100kHz']={
            'RegisterControl'   : "config/UhrWaveformControlAxi_registers_100kHz.yml",
            'TriggerReg'        : "config/UhrTrigControlAxi_registers_100kHz.yml",
            'SACIReg'           : "config/ePixUHR_SACI_Registers.yml",
            'FramerReg'         : "config/UhrFramerAxi_registers.yml",
            'General'           : "config/ePixUHR_camera_general_settings.yml",
        }
ymlfilename['_1MHz']={
            'RegisterControl'   : "config/UhrWaveformControlAxi_registers_1MHz.yml",
            'TriggerReg'        : "config/UhrTrigControlAxi_registers_1MHz.yml",
            'SACIReg'           : "config/ePixUHR_SACI_Registers.yml",
            'FramerReg'         : "config/UhrFramerAxi_registers.yml",
            'General'           : "config/ePixUHR_camera_general_settings.yml",
}
ymlfilename['temp']={
            'RegisterControl'   : "config/UhrWaveformControlAxi_registers_temp.yml",
            'TriggerReg'        : "config/UhrTrigControlAxi_registers_temp.yml",
            'SACIReg'           : "config/ePixUHR_SACI_Registers.yml",
            'FramerReg'         : "config/UhrFramerAxi_registers.yml",
            'General'           : "config/ePixUHR_camera_general_settings.yml",
}
        
ymlfilename['MHzmode']={
            'RegisterControl'   : "config/UhrWaveformControlAxi_registers_MHz_Mode.yml",
            'TriggerReg'        : "config/UhrTrigControlAxi_registers_MHz_Mode.yml",
            'SACIReg'           : "config/ePixUHR_SACI_Registers_MHz_mode.yml",
            'FramerReg'         : "config/UhrFramerAxi_registers_MHz_mode.yml",
            'General'           : "config/ePixUHR_camera_general_settings.yml",
}

def recursive_list(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_list(value)
        else:
            yield (key, value)


def epixUHR_cdict():

    top = cdict()
    d={}
    
    yamlpath="/cds/home/m/melchior/git/epix-uhr-dev/software/"
    mode = '_100kHz'
    #for mode in ymlfilename.keys():
    dmodefilename={}
    sorted_dict={}
    for filename in ymlfilename[mode]:
        sorted_list=[]
        dmode={}
        with open(yamlpath+ymlfilename[mode][filename], 'r') as file:
            prime_service = yaml.safe_load(file)
            
        for key, value in recursive_list(prime_service):
            if key != 'enable': sorted_list.append(key)
            
        sorted_dict[filename]=','.join(sorted_list)
        dmode.update(prime_service['Root']['App'])
        
        
        dmodefilename[filename]=dmode
        
    d=dmodefilename
    d['sorted']=sorted_dict
    
    top.init_from_dict(d, base=f'expert.App')
    #top.define_enum('ymlenum', {'_35kHz':0, '_100kHz':1, '_1MHz':2, 'temp':3, 'MHzmode':4})
    
    #top.set('expert.App', 1, 'ymlenum')

    #top.setAlg('config', [2,0,0])
    pixelMap = np.zeros((elemRows*2,elemCols*2),dtype=np.uint8)
    top.set("user.pixel_map", pixelMap)
    
    top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- user interface --"
    help_str += "\nstart_ns     : exposure start (nanoseconds)"
    help_str += "\ngate_ns     : exposure time (nanoseconds)"
    top.set("help.user:RO", help_str, 'CHARSTR')

    # set to 88000 to get triggerDelay larger than zero when
    # L0Delay is 81 (used by TMO)
    top.set("user.start_ns" , 88000, 'UINT32') # taken from epixHR
    #top.set("user.gate_ns" , 154000, 'UINT32') # taken from lcls1 xpptut15 run 260
    #add daqtriggerdelay and runtriggerdelay?
    
    top.set("user.run_trigger_group", 6, 'UINT32')
    
    return top

if __name__ == "__main__":
    create = False# True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args
    
    db = 'configdb' if args.prod else 'devconfigdb'
    
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,    
                        root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('epixuhr')

    top = epixUHR_cdict()
    top.setInfo('epixuhr', args.name, args.segm, args.id, 'No comment')

    mycdb.modify_device(args.alias, top)


from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import pyrogue
import pyrogue.interfaces.simulation
import epix100a_gen2
import ePixFpga as fpga
import numpy as np
import sys
import IPython
import argparse
import re

mem = pyrogue.interfaces.simulation.MemEmulate()

class EpixBoard(pyrogue.Root):
    def __init__(self):
        super().__init__(name = 'ePixBoard', description = 'ePix 100a Board')
        self.add(fpga.Epix100a(name='ePix100aFPGA', memBase=mem))

def get_configuration_values(key, value, current_entry_path, entry_list):
    if isinstance(value, dict):
        current_entry_path.append(key)
        for key, value in value.items():
            get_configuration_values(key, value, current_entry_path[:], entry_list)
    else:
        current_entry_path.append(key)
        entry_list.append((current_entry_path, value))

def retrieve_rogue_information(entry_path, tree_dict):
    value = tree_dict
    for branch in entry_path:
        value = value[branch]
    return value

def epix100_cdict(data_from_yaml):
    root = EpixBoard()

    with EpixBoard() as root:
        tree_dict = root.treeDict()
    
    for entry in ("runControl", "dataWriter"):
        if entry in data_from_yaml["ePixBoard"]:
            del(data_from_yaml["ePixBoard"][entry]) 
   
    current_entry_path = []
    entry_list = []
    for key, value in data_from_yaml.items():
        get_configuration_values(key, value, current_entry_path, entry_list)

    type_conversion_dict = {
        "bool": "boolEnum",
        "uint1": "UINT8",
        "uint2": "UINT8",
        "uint3": "UINT8",
        "uint4": "UINT8",
        "uint5": "UINT8",
        "uint6": "UINT8",
        "uint7": "UINT8",
        "uint8": "UINT8",
        "uint10": "UINT16",
        "uint13": "UINT16",
        "uint16": "UINT16",
        "uint31": "UINT32",
        "uint32": "UINT32",
    }

    top = cdict()
    top.setAlg('config', [2,0,0])
    top.define_enum('boolEnum', {'False':0, 'True':1})
    
    for entry in entry_list:
        temp_path=f"expert.{'.'.join(entry[0][1:])}"
        configdb_path = re.sub("\[(.*?)\]", "_\\1", temp_path)
        info = retrieve_rogue_information(entry[0], tree_dict)
        if info['class'] == 'LinkVariable':
            print(f"Skipping {configdb_path}: Linked Variable")
            continue
        original_type = info["typeStr"].lower()
        original_value = entry[1]
        if info["ndType"] != None:
            array_type = info["ndType"]
            temp_configdb_value = [entry for entry in original_value.split("[")[1].split("]")[0].split(",")]
            if "x" in temp_configdb_value[0]:
                num_base = 16
            else:
                num_base = 10
            temp_configdb_value2 = [int(entry, base=num_base) for entry in temp_configdb_value]
            configdb_type = type_conversion_dict[str(array_type).lower()]
            configdb_value = list(np.array(temp_configdb_value2, dtype=array_type))
        else: 
            if original_type in type_conversion_dict:
                configdb_type = type_conversion_dict[original_type]
            else:
                print(f"Type unknown, please define the psana type: {original_type}")
                print(info)
                sys.exit(0)
            if info["enum"] != None:
                if configdb_type == "boolEnum":
                    if original_value == True:
                         configdb_value = 1
                    elif original_value == False:
                        configdb_value = 0
                elif original_value in info["enum"].values():
                    dict_keys = list(info["enum"].keys())
                    dict_values = list(info["enum"].values())
                    index = dict_values.index(original_value)
                    configdb_value = dict_keys[index]
            else:
                configdb_value = original_value
        print(f"Writing into database: {configdb_path}, {configdb_value}, {configdb_type}") 
        top.set(f"{configdb_path}" , configdb_value, f"{configdb_type}") # taken from epixHR

    #top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    #top.set("firmwareVersion:RO",   0, 'UINT32')

    #help_str  = "-- user interface --"
    #help_str += "\nstart_ns     : exposure start (nanoseconds)"
    #help_str += "\ngate_ns     : exposure time (nanoseconds)"
    #top.set("help.user:RO", help_str, 'CHARSTR')

    # set to 88000 to get triggerDelay larger than zero when
    # L0Delay is 81 (used by TMO)
    top.set("user.start_ns" , 88000, 'UINT32') # taken from epixHR
    top.set("user.gate_ns" , 154000, 'UINT32') # taken from lcls1 xpptut15 run 260
    # add daqtriggerdelay and runtriggerdelay?

    # timing system
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.PauseThreshold',16,'UINT32')
    top.set('expert.cfgyaml:RO','NoYaml','CHARSTR')

    return top

if __name__ == "__main__":
    create = True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args

    print(f"Load configuration from: {args.yaml}")
    data_from_yaml = pyrogue.yamlToData(fName=args.yaml)
    #print(f'keys {d}')

    db = 'configdb' if args.prod else 'devconfigdb'
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('epix100')

    top = epix100_cdict(data_from_yaml=data_from_yaml)
    top.setInfo('epix100', args.name, args.segm, args.id, 'No comment')

    mycdb.modify_device(args.alias, top)

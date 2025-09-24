#!/usr/bin/env python

"""configdb_multimodifier.py: Performs multiple modifications of a variable over several detectors in configdb. configdb_mult
imodifier.py uses configdb.py"""

__author__      = "Riccardo Melchiorri"


from psdaq.configdb.configdb import configdb
import os
import numpy as np

def nested_set(dic, keys, value):
    d = dic
    for key in keys[:-1]:
        if key in d:
            d = d[key]
        else:
            return dic
    if keys[-1] in d:
        d[keys[-1]] = value
    return dic

def configdb_multimod(URI_CONFIGDB = 'https://pswww.slac.stanford.edu/ws-kerb/configdb/ws', DEV = 'BEAM',ROOT_CONFIGDB = 'configDB', INSTRUMENT = 'tmo', DETECTOR=['hsd_0'] , CONFIG_KEY=['user','raw','prescale'], CONFIG_VALUE=1, MODIFY=False):
    
    sout=[]
    if "ws-auth" in URI_CONFIGDB:
        confdb = configdb(URI_CONFIGDB, INSTRUMENT, create=False, root=ROOT_CONFIGDB, user=f"{INSTRUMENT}opr")
    else:
        confdb = configdb(URI_CONFIGDB, INSTRUMENT, create=False, root=ROOT_CONFIGDB )
    
    for det in DETECTOR:
        
        config = confdb.get_configuration(DEV, f'{det}', hutch=INSTRUMENT)
        con = config
        for k in CONFIG_KEY:
            con=con[k]
        print(f'{det}:{CONFIG_KEY}:{con} original')   
        sout.append(f'{det}:{CONFIG_KEY}:{con} original')
        if MODIFY:
            
            if (f'{CONFIG_VALUE}'.startswith("delta:")):
                print("delta detected")
                print(type(con))
                CONFIG_VALUE_=con+type(con)(CONFIG_VALUE.split(":")[1])
            else:
                print("delta NOT detected")
                CONFIG_VALUE_=type(con)(CONFIG_VALUE)   
            #CONFIG_VALUE_=type(con)(CONFIG_VALUE_)   

#need to be the same type that is written in the database otherwise the comparison fails

            if con is not CONFIG_VALUE_:
                config=nested_set(config, CONFIG_KEY, CONFIG_VALUE_)
                con = config
                for k in CONFIG_KEY:
                    con=con[k]
                print(f'{det}:{CONFIG_KEY}:{con} modified')
                sout.append(f'{det}:{CONFIG_KEY}:{con} modified')
                
                try:
                    new_key = confdb.modify_device(DEV, config, hutch=INSTRUMENT)
                except:
                    print("There has been an erro wrint the database")
    return sout


def main():
    import argparse
    # create the top-level parser
    parser = argparse.ArgumentParser(description='configuration database CLI')
    parser.add_argument('--URI_CONFIGDB', default='https://pswww.slac.stanford.edu/ws-kerb/configdb/ws/',
                        help='configuration database connection')
    parser.add_argument('--ROOT_CONFIGDB', default='configDB', help='configuration database root (default: configDB)')
    parser.add_argument('--DEV', default='BEAM', help='configuration database dev (default: BEAM)')
    parser.add_argument('--INSTRUMENT', default='tmo', help='configuration database instrument (default: tmo)')
    parser.add_argument('--DETECTOR', default='hsd_1', help='configuration database detector (default: hsd_1)')
    parser.add_argument('--CONFIG_KEY', default='["user"]', help='configuration database CONFIG_KEY (default: ["user"])')
    parser.add_argument('--CONFIG_VALUE', default=0, help='configuration database CONFIG_VALUE (default: 0)')
    parser.add_argument('--MODIFY', default=False, help='configuration database MODIFY (default: False)')
    subparsers = parser.add_subparsers()


    # parse the args and call whatever function was selected
    args = parser.parse_args()
    try:
        subcommand = args.func
    except Exception:
        parser.print_help(sys.stderr)
        sys.exit(1)
    try:
        subcommand(args)
    except Exception as ex:
        sys.exit(ex)



if __name__ == "__main__":
    main()    
#dbmultimodifier(URI_CONFIGDB = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws', DEV = 'BEAM', ROOT_CONFIGDB = 'configDB', INSTRUMENT = 'tmo', DETECTOR='hsd_',  CONFIG_KEY=['user','raw','prescale'], CONFIG_VALUE=0, MODIFY=False)



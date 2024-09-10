#!/usr/bin/env python

"""configdb_multimodifier.py: Performs multiple modifications of a variable over several detectors in configdb. configdb_multimodifier.py uses configdb.py"""

__author__      = "Riccardo Melchiorri"


from psdaq.configdb.configdb import configdb
import os

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

def dbmultimodifier(URI_CONFIGDB = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws', ROOT_CONFIGDB = 'configDB', INSTRUMENT = 'tmo', USER='tmoopr', DETECTOR=['hsd_0'] , CONFIG_KEY=['user','raw','prescale'], CONFIG_VALUE=1, MODIFY=False):
    
    confdb = configdb(URI_CONFIGDB, INSTRUMENT, create=False, root=ROOT_CONFIGDB, user=USER, password=os.getenv('CONFIGDB_AUTH'))
    
    for det in DETECTOR:
        config = confdb.get_configuration('BEAM', f'{det}', hutch=INSTRUMENT)
        con = config
        for k in CONFIG_KEY:
            con=con[k]
        print(f'{det}:{CONFIG_KEY}:{con} original')            
        if MODIFY:
            CONFIG_VALUE=type(con)(CONFIG_VALUE)   #need to be the same type that is written in the database otherwise the comparison fails

            if con is not CONFIG_VALUE:
                config=nested_set(config, CONFIG_KEY, CONFIG_VALUE)
                con = config
                for k in CONFIG_KEY:
                    con=con[k]
                print(f'{det}:{CONFIG_KEY}:{con} modified')
                #new_key = confdb.modify_device('BEAM', config, hutch=INSTRUMENT)

if __name__ == "__main__":
    dbmultimodifier(URI_CONFIGDB = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws', ROOT_CONFIGDB = 'configDB', INSTRUMENT = 'tmo', USER='tmoopr', DETECTOR='hsd_', DRANGE=range(20), CONFIG_KEY=['user','raw','prescale'], CONFIG_VALUE=0, MODIFY=False)




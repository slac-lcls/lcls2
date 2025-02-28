
from psdaq.configdb.configdb import configdb
import json

def main():
    data={}
    data["Root"]   = 'configDB'
    data["Inst"]   = 'TMO'
    detType={}
    
    for d in ['https://pswww.slac.stanford.edu/ws-auth/configdb/ws', 'https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws' ] : 
    
        confdb = configdb(d, data['Inst'], create=False, root=data['Root'])
        list_of_hutch_names = confdb.get_hutches() # ['TMO', 'CXI', etc.]
        list_of_device_configuration_names = confdb.get_device_configs() #['test']  
        print(f'url:{d}')
        for myhutch in list_of_hutch_names:
            print(f'hutch: {myhutch}')
            list_of_alias_names = confdb.get_aliases(hutch=myhutch)
            #print(list_of_alias_names)
            for myalias in list_of_alias_names:
                print(f'alias: {myalias}')
                list_of_device_names = confdb.get_devices(myalias, hutch=myhutch) 
                #print(list_of_device_names)
                for dev in list_of_device_names:
                    print(f'dev: {dev}')
                    #print(f'{myalias} {dev} {myhutch}')
                    try:
                        config = confdb.get_configuration( myalias, dev, hutch=myhutch)
                        detType[f'{d}:{myhutch}:{myalias}:{dev}'] = config['detType:RO']
                    except:
                        detType[f'{d}:{myhutch}:{myalias}:{dev}'] = '' 
    json_string = json.dumps(detType)
    print(json_string)
    

if __name__ == "__main__":
    main()

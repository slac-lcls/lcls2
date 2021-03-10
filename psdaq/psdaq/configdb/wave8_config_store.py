from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import sys
import argparse
import IPython
import pyrogue as pr

def lookupValue(d,name):
    key = name.split('.',1)
    if key[0] in d:
        v = d[key[0]]
        if isinstance(v,dict):
            return lookupValue(v,key[1])
        elif isinstance(v,bool):
            return 1 if v else 0
        else:
            return v
    else:
        return None
            
class mcdict(cdict):
    def __init__(self, fn=None):
        super().__init__(self)

        self._yamld = {}
        if fn:
            print('Loading yaml...')
            self._yamld = pr.yamlToData(fName=fn)

    #  intercept the set call to replace value with yaml definition
    def init(self, prefix, name, value, type="INT32", override=False, append=False):
        v = lookupValue(self._yamld,name)
        if v:
            print('Replace {:}[{:}] with [{:}]'.format(name,value,v))  
            value = v
        self.set(prefix+':RO.'+name+':RO', value, type, override, append)
    
def write_to_daq_config_db(args):

    #database contains collections which are sets of documents (aka json objects).
    #each type of device has a collection.  The elements of that collection are configurations of that type of device.
    #e.g. there will be OPAL, EVR, and YUNGFRAU will be collections.  How they are configured will be a document contained within that collection
    #Each hutch is also a collection.  Documents contained within these collection have an index, alias, and list of devices with configuration IDs
    #How is the configuration of a state is described by the hutch and the alias found.  E.g. TMO and BEAM.  TMO is a collection.  
    #BEAM is an alias of some of the documents in that collection. The document with the matching alias and largest index is the current 
    #configuration for that hutch and alias.  
    #When a device is configured, the device has a unique name OPAL7.  Need to search through document for one that has an NAME called OPAL7.  This will have
    #have two fields "collection" and ID field (note how collection here is a field. ID points to a unique document).  This collection field and 
    #ID point to the actuall Mongo DB collection and document

    create = True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    db   = 'configdb' if args.prod else 'devconfigdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('wave8')
    
    top = mcdict(args.yaml)
    top.setInfo('wave8', args.name, args.segm, args.id, 'No comment')
    top.setAlg('config', [0,1,0])

    help_str = "-- user. --"
    help_str += "\ndelta_ns  : nanoseconds difference from LCLS-1 timing"
    top.set("help:RO", help_str, 'CHARSTR')

    #  Split configuration into two sections { User and Expert }
    #  Expert configuration is the basis, and User configuration overrides 
    #  Expert variables map directly to Rogue variables

    top.set("user.delta_ns"                ,-807692,'INT32')    # [ns from LCLS1 timing]

    top.set("expert.TriggerEventManager.TriggerEventBuffer.TriggerDelay"  , 0,'UINT32')  # user config

    mycdb.add_alias(args.alias)
    mycdb.modify_device(args.alias, top)


if __name__ == "__main__":
    args = cdb.createArgs().args
    write_to_daq_config_db(args)

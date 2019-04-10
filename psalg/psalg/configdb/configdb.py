from pymongo import *
from .typed_json import cdict
import datetime
import time, re, sys

# Note: This was originally written with mongodb transactions in mind, 
# which is why we are passing "session" all over the place.  However,
# these are really only used in three places:
#     add_alias: When we need to get a new key and then insert a new
#                record with this key.
#     modify_device: When we save a bunch of configurations, then
#                get a new key and insert a new record with this key.
#     transfer_config: After retrieving the desired configuration, we
#                get a new key and insert a new record with this key.
# Now in modify_device, it is annoying but harmless to save configurations
# and then have other saves or the insert potentially fail, leaving a
# bunch of unused configurations in the database, but we can live with this.
# 
# In these three functions, I have changed:
#     with self.client.start_session() as session:
#         with session.start_transaction():
# to:
#     if True:
#             session = None
# (Yes, the indenting is ugly, but just in case we ever decide to go back.)
#
# get_key changes to do the counter magic as well.

class configdb(object):
    client = None
    cdb = None
    cfg_coll = None

    # Parameters:
    #     server - The MongoDB string: "user:password@host:port" or 
    #              "host:port" if no authentication.
    #     h      - The current hutch name.
    #     root   - The root database name, usually "configDB"
    #     drop   - If True, the root database will be dropped.
    #     create - If True, try to create the database and collections
    #              for the hutch, device configurations, and counters.
    def __init__(self, server, h=None, create=False, root="configDB"):
        if self.client == None:
            self.client = MongoClient("mongodb://" + server)
            self.cdb = self.client.get_database(root)
            self.cfg_coll = self.cdb.device_configurations
            if create:
                try:
                    self.cdb.create_collection("device_configurations")
                except:
                    pass
                try:
                    self.cdb.create_collection("counters")
                except:
                    pass
            self.set_hutch(h, create=create)

    # Change to the specified hutch, creating it if necessary.
    def set_hutch(self, h, create=False):
        self.hutch = h
        if h is None:
            self.hutch_coll = None
        else:
            self.hutch_coll = self.cdb[h]
        if create and h is not None:
            try:
                self.cdb.create_collection(h)
            except:
                pass
            try:
                if not self.cdb.counters.find_one({'hutch': h}):
                    self.cdb.counters.insert_one({'hutch': h, 'seq': -1})
            except:
                pass

    # Return the highest key for the specified alias, or highest + 1 for all
    # aliases in the hutch if not specified.
    def get_key(self, alias=None, hutch=None, session=None):
        if hutch is None:
            hutch = self.hutch
        try:
            if isinstance(alias, str) or (sys.version_info.major == 2 and
                                          isinstance(alias, unicode)):
                d = self.cdb[hutch].find({'alias' : alias}, session=session).sort('key', DESCENDING).limit(1)[0]
                return d['key']
            else:
                d = self.cdb.counters.find_one_and_update({'hutch': hutch},
                                                          {'$inc': {'seq': 1}},
                                                          session=session,
                                                          return_document=ReturnDocument.AFTER)
                return d['seq']
        except:
            return None

    # Return the current entry (with the highest key) for the specified alias.
    def get_current(self, alias, hutch=None, session=None):
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        try:
            return hc.find({"alias": alias}, session=session).sort('key', DESCENDING).limit(1)[0]
        except:
            return None

    # Create a new alias in the hutch, if it doesn't already exist.
    def add_alias(self, alias):
        if True:
                session = None
                if self.hutch_coll.find_one({'alias': alias},
                                            session=session) is None:
                    kn = self.get_key(session=session)
                    self.hutch_coll.insert_one({
                        "date": datetime.datetime.utcnow(),
                        "alias": alias, "key": kn,
                        "devices": []}, session=session)

    # Create a new device_configuration if it doesn't already exist!
    def add_device_config(self, cfg, session=None):
        # Validate name?
        if self.cdb[cfg].count_documents({}) != 0:
            return
        try:
            self.cdb.create_collection(cfg)
        except:
            pass
        self.cdb[cfg].insert_one({'config': {}}, session=session)
        self.cfg_coll.insert_one({'collection': cfg}, session=session)

    # Save a device configuration and return an object ID.  Try to find it if 
    # it already exists! Value should be a typed json dictionary.
    def save_device_config(self, cfg, value, session=None):
        if self.cdb[cfg].count_documents({}, session=session) == 0:
            raise NameError("save_device_config: No documents found for %s." % cfg)
        try:
            d = self.cdb[cfg].find_one({'config': value}, session=session)
            return d['_id']
        except:
            pass
        try:
            r = self.cdb[cfg].insert_one({'config': value}, session=session)
            return r.inserted_id
        except:
            return None

    # Modify the current configuration for a specific device, adding it if
    # necessary.  name is the device and value is a json dictionary for the 
    # configuration.  Return the new configuration key if successful and 
    # raise an error if we fail.
    def modify_device(self, alias, value, hutch=None):
        device = value.get('detName')
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        c = self.get_current(alias, hutch)
        if c is None:
            raise NameError("modify_device: %s is not a configuration name!"
                            % alias)
        if isinstance(value, cdict):
            value = value.typed_json()
        if not isinstance(value, dict):
            raise TypeError("modify_device: value is not a dictionary!")
        if not "detType" in value.keys():
            raise ValueError("modify_device: value has no detType set!")
        if True:
                session = None
                collection = value["detType"]
                cfg = {'_id': self.save_device_config(collection, 
                                                      value, session),
                       'collection': collection}
                del c['_id']
                for l in c['devices']:
                    if l['device'] == device:
                        if l['configs'] == [cfg]:
                            raise ValueError("modify_device: No change!")
                        c['devices'].remove(l)
                        break
                kn = self.get_key(session=session, hutch=hutch)
                c['key'] = kn
                c['devices'].append({'device': device, 'configs': [cfg]})
                c['devices'].sort(key=lambda x: x['device'])
                c['date'] = datetime.datetime.utcnow()
                hc.insert_one(c, session=session)
        return kn
    
    # Retrieve the configuration of the device with the specified key or alias.
    # This returns a dictionary where the keys are the collection names and the 
    # values are typed JSON objects representing the device configuration(s).
    def get_configuration(self, key_or_alias, device, hutch=None):
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        if isinstance(key_or_alias, str) or (sys.version_info.major == 2 and
                                             isinstance(key_or_alias, unicode)):
            key = self.get_key(key_or_alias, hutch)
            if key is None:
                return None
        else:
            key = key_or_alias
        #try:
        if True:
            c = hc.find_one({"key": key})
            cfg = None
            for l in c["devices"]:
                if l['device'] == device:
                    cfg = l['configs']
                    break
            if cfg is None:
                raise ValueError("get_configuration: No device %s!" % device)
            cname = cfg[0]['collection']
            r = self.cdb[cname].find_one({"_id" : cfg[0]['_id']})
            return r['config']
        #except:
        #    return None

    # Return a list of all hutches.
    def get_hutches(self):
        return [v['hutch'] for v in self.cdb.counters.find()]

    # Return a list of all aliases in the hutch.
    def get_aliases(self, hutch=None):
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        return [v['_id'] for  v in hc.aggregate([{"$group": 
                                                  {"_id" : "$alias"}}])] 

    # Return a list of all device configurations.
    def get_device_configs(self):
        return [v['collection'] for v in self.cfg_coll.find()]

    # Return a list of all devices in an alias/hutch.
    def get_devices(self, key_or_alias, hutch=None):
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        if isinstance(key_or_alias, str) or (sys.version_info.major == 2 and
                                             isinstance(key_or_alias, unicode)):
            key = self.get_key(key_or_alias, hutch)
            if key is None:
                return None
        else:
            key = key_or_alias
        c = hc.find_one({"key": key})
        return [l['device'] for l in c["devices"]]

    # Print all of the configurations for the hutch.
    def print_configs(self, hutch=None):
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        for v in hc.find():
            print(v)

    # Print all of the device configurations, or all of the configurations 
    # for a specified device.
    def print_device_configs(self, name="device_configurations"):
        for v in self.cdb[name].find():
            print(v)

    # Transfer a configuration from another hutch to the current hutch,
    # returning the new key.
    def transfer_config(self, oldhutch, oldalias, olddevice, newalias,
                        newdevice):
        k = self.get_key(oldalias, oldhutch)
        pipeline = [
            {"$unwind": "$devices"},
            {"$match": {'key': k, 'devices.device': olddevice}}
        ]
        cfg = next(self.cdb[oldhutch].aggregate(pipeline))['devices']['configs']
        cnew = self.get_current(newalias)
        if True:
                session = None
                kn = self.get_key(session=session)
                cnew['key'] = kn
                del cnew['_id']
                for l in cnew['devices']:
                    if l['device'] == newdevice:
                        if l['configs'][0]['collection'] != cfg[0]['collection']:
                            raise ValueError("transfer_config: Different collections!")
                        if l['configs'] == cfgs:
                            raise ValueError("transfer_config: No change!")
                        cnew['devices'].remove(l)
                        break
                cnew['devices'].append({'device': newdevice, 'configs': cfg})
                cnew['devices'].sort(key=lambda x: x['device'])
                cnew['date'] = datetime.datetime.utcnow()
                self.hutch_coll.insert_one(cnew, session=session)
        return kn

    # Get the history of the device configuration for the variables 
    # in plist.  The variables are dot-separated names with the first
    # component being the the device configuration name.
    def get_history(self, alias, device, plist, hutch=None):
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        pipeline = [
            {"$unwind": "$devices"},
            {"$match": {'alias': alias, 'devices.device': device}},
            {"$sort":  {'key': ASCENDING}}
        ]
        l = []
        for c in list(hc.aggregate(pipeline)):
            d = {'date': c['date'], 'key': c['key']}
            cfg = c['devices']['configs'][0]
            r = self.cdb[cfg['collection']].find_one({"_id" : cfg["_id"]})
            cl = cdict(r['config'])
            for p in plist:
                d[p] = cl.get(p)
            l.append(d)
        return l

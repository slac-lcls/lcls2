from pymongo import *
from psana.dgramPort.typed_json import cdict
import datetime
import time, re, sys

class configdb(object):
    client = None
    cdb = None
    cfg_coll = None

    def __init__(self, server, h, create=False):
        if self.client == None:
            self.client = MongoClient("mongodb://" + server)
            self.cdb = self.client.get_database("configDB")
            self.cfg_coll = self.cdb.device_configurations
            if create:
                try:
                    self.cdb.create_collection("device_configurations")
                except:
                    pass
            self.hutch = h
            self.hutch_coll = self.cdb[h]
            if create:
                try:
                    self.cdb.create_collection(h)
                except:
                    pass

    # Return the highest key for the specified alias, or highest + 1 for all aliases
    # in the hutch if not specified.
    def get_key(self, alias=None, hutch=None, session=None):
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        if isinstance(alias, str) or (sys.version_info.major == 2 and isinstance(alias, unicode)):
            a = {"alias": alias}
            inc = 0
        else:
            a = {}
            inc = 1
        try:
            return hc.find(a, session=session).sort('key', DESCENDING).limit(1)[0]['key'] + inc
        except:
            if alias is None:
                return 0
            else:
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
        with self.client.start_session() as session:
            with session.start_transaction():
                if self.hutch_coll.find_one({'alias': alias}, session=session) is None:
                    kn = self.get_key(session=session)
                    self.hutch_coll.insert_one({"date": datetime.datetime.utcnow(),
                                           "alias": alias, "key": kn,
                                           "devices": []},
                                          session=session)

    # Create a new device_configuration if it doesn't already exist!
    def add_device_config(self, cfg, session=None):
        # Validate name?
        if self.cdb[cfg].count_documents({}) != 0:
            return
        try:
            self.cdb.create_collection(cfg)
        except:
            pass
        self.cdb[cfg].insert_one({'config': {}}, session=session)   # Make an empty config!
        self.cfg_coll.insert_one({'collection': cfg}, session=session)

    # Save a device configuration and return an object ID.  Try to find it if it already exists!
    # Value should be a typed json dictionary.
    def save_device_config(self, cfg, value, session=None):
        if self.cdb[cfg].count_documents({}, session=session) == 0:
            raise NameError("save_device_config: No device configuration %s" % name)
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

    # Modify the current configuration for a specific device, adding it if necessary.
    # name is the device and value is a dictionary for the configuration.  Top level
    # keys must be device configuration names and the values should be typed json
    # dictionaries.  Return the new configuration key if successful and raise an error
    # if we fail.
    def modify_device(self, alias, device, value, hutch=None):
        if hutch is None:
            hc = self.hutch_coll
        else:
            hc = self.cdb[hutch]
        c = self.get_current(alias, hutch)
        if c is None:
            raise NameError("modify_device: %s is not a configuration name!" % alias)
        if not isinstance(value, dict):
            raise TypeError("modify_device: value is not a dictionary!")
        cfgs = []
        with self.client.start_session() as session:
            with session.start_transaction():
                for k in sorted(value.keys()):
                    v = value[k]
                    if isinstance(v, cdict):
                        v = v.typed_json()
                    try:
                        cfgs.append({'_id': self.save_device_config(k, v, session), 'collection': k})
                    except:
                        raise
                kn = self.get_key(session=session, hutch=hutch)
                c['key'] = kn
                del c['_id']
                for l in c['devices']:
                    if l['device'] == device:
                        if l['configs'] == cfgs:
                            raise ValueError("modify_device: No change!")
                        c['devices'].remove(l)
                        break
                c['devices'].append({'device': device, 'configs': cfgs})
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
        if isinstance(key_or_alias, str) or (sys.version_info.major == 2 and isinstance(key_or_alias, unicode)):
            key = self.get_key(key_or_alias, hutch)
            if key is None:
                return None
        else:
            key = key_or_alias
        try:
            c = hc.find_one({"key": key})
            clist = None
            for l in c["devices"]:
                if l['device'] == device:
                    clist = l['configs']
                    break
            if clist is None:
                raise ValueError("get_configuration: No device %s!" % device)
            d = {}
            for o in clist:
                cname = o['collection']
                r = self.cdb[cname].find_one({"_id" : o['_id']})
                d[cname] = r['config']
            return d
        except:
            return None

    # Return a list of all aliases in the hutch.
    def get_aliases(self):
        return [v['_id'] for  v in self.hutch_coll.aggregate([{"$group": {"_id" : "$alias"}}])] 

    # Print all of the configurations for the hutch.
    def print_configs(self):
        for v in self.hutch_coll.find():
            print(v)

    # Print all of the device configurations, or all of the configurations for a specified device.
    def print_device_configs(self, name="device_configurations"):
        for v in self.cdb[name].find():
            print(v)

    # Transfer a configuration from another hutch to the current hutch, returning the new key.
    def transfer_config(self, oldhutch, oldalias, olddevice, newalias, newdevice):
        k = self.get_key(oldalias, oldhutch)
        pipeline = [
            {"$unwind": "$devices"},
            {"$match": {'key': k, 'devices.device': olddevice}}
        ]
        cfgs = next(self.cdb[oldhutch].aggregate(pipeline))['devices']['configs']
        cnew = self.get_current(newalias)
        with self.client.start_session() as session:
            with session.start_transaction():
                kn = self.get_key(session=session)
                cnew['key'] = kn
                del cnew['_id']
                for l in cnew['devices']:
                    if l['device'] == newdevice:
                        if l['configs'] == cfgs:
                            raise ValueError("transfer_config: No change!")
                        cnew['devices'].remove(l)
                        break
                cnew['devices'].append({'device': newdevice, 'configs': cfgs})
                cnew['devices'].sort(key=lambda x: x['device'])
                cnew['date'] = datetime.datetime.utcnow()
                self.hutch_coll.insert_one(cnew, session=session)
        return kn

    # Get the history of the device configuration for the variables 
    # in plist.  The variables are dot-separated names with the first
    # component being the the device configuration name.
    def get_history(self, alias, device, plist):
        pipeline = [
            {"$unwind": "$devices"},
            {"$match": {'alias': alias, 'devices.device': device}},
            {"$sort":  {'key': ASCENDING}}
        ]
        l = []
        pl = []
        for p in plist:
            m = re.search('^([^.]*)[.](.*)$', p)
            if m:
                pl.append((m.group(1), m.group(2)))
            else:
                raise ValueError("%s is not a dot-separated name!" % p)
        dc = set([x[0] for x in pl])
        for c in list(self.hutch_coll.aggregate(pipeline)):
            d = {'date': c['date'], 'key': c['key']}
            for o in c['devices']['configs']:
                cname = o['collection']
                if cname in dc:
                    r = self.cdb[cname].find_one({"_id" : o['_id']})
                    cl = cdict(r['config'])
                    for p in pl:
                        if cname == p[0]:
                            d[p[0]+"."+p[1]] = cl.get(p[1])
            l.append(d)
        return l

import requests
import json
import logging
from .typed_json import cdict

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, float) and not math.isfinite(o):
            return str(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

class configdb(object):

    # Parameters:
    #     url    - e.g. "http://localhost:5000/ws"
    #     hutch  - Instrument name, e.g. "tmo"
    #     create - If True, try to create the database and collections
    #              for the hutch, device configurations, and counters.
    #     root   - Database name, usually "configDB"
    def __init__(self, url, hutch, create=False, root="NONE"):
        if root == "NONE":
            raise Exception("configdb: Must specify root!")
        self.hutch  = hutch
        self.prefix = url.strip('/') + '/' + root + '/'
        self.timeout = 3.05     # timeout for http requests

        if create:
            try:
                requests.get(self.prefix + 'create_collections/' + hutch + '/',
                             timeout=self.timeout)
            except requests.exceptions.RequestException as ex:
                logging.error('Web server error: %s' % ex)

    # Retrieve the configuration of the device with the specified alias.
    # This returns a dictionary where the keys are the collection names and the 
    # values are typed JSON objects representing the device configuration(s).
    # On error return an empty dictionary.
    def get_configuration(self, alias, device, hutch=None):
        if hutch is None:
            hutch = self.hutch
        try:
            xx = requests.get(self.prefix + 'get_configuration/' + hutch + '/' +
                              alias + '/' + device + '/',
                              timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return dict()

        if not xx['success']:
            logging.error('%s' % xx['msg'])
            return dict()
        else:
            return xx['value']

    # Get the history of the device configuration for the variables 
    # in plist.  The variables are dot-separated names with the first
    # component being the the device configuration name.
    def get_history(self, alias, device, plist, hutch=None):
        if hutch is None:
            hutch = self.hutch
        value = JSONEncoder().encode(plist)
        try:
            xx = requests.get(self.prefix + 'get_history/' + hutch + '/' +
                              alias + '/' + device + '/',
                              json=value,
                              timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            xx = []

        # try to clean up superfluous keys from serialization
        if 'value' in xx:
            try:
                for item in xx['value']:
                    bad_keys = []
                    for kk in item.keys():
                        if not kk.isalnum():
                            bad_keys += kk
                    for bb in bad_keys:
                        item.pop(bb, None)
            except Exception:
                pass

        return xx

    # Return version as a dictionary.
    # On error return an empty dictionary.
    def get_version(self):
        try:
            xx = requests.get(self.prefix + 'get_version/',
                              timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return {}
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
        return xx['value']

    # Return the highest key for the specified alias, or highest + 1 for all
    # aliases in the hutch if not specified.
    # On error return an empty list.
    def get_key(self, alias=None, hutch=None, session=None):
        if hutch is None:
            hutch = self.hutch
        try:
            if alias is None:
                xx = requests.get(self.prefix + 'get_key/' + hutch + '/',
                                  timeout=self.timeout).json()
            else:
                xx = requests.get(self.prefix + 'get_key/' + hutch + '/' +
                                  '?alias=%s' % alias,
                                  timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
        return xx['value']

    # Return a list of all hutches available in the config db.
    # On error return an empty list.
    def get_hutches(self):
        try:
            xx = requests.get(self.prefix + 'get_hutches/',
                              timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])

        return xx['value']

    # Return a list of all aliases in the hutch.
    # On error return an empty list.
    def get_aliases(self, hutch=None):
        if hutch is None:
            hutch = self.hutch
        try:
            xx = requests.get(self.prefix + 'get_aliases/' + hutch + '/',
                              timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])

        return xx['value']

    # Create a new alias in the hutch, if it doesn't already exist.
    def add_alias(self, alias):
        try:
            xx = requests.get(self.prefix + 'add_alias/' + self.hutch + '/' +
                              alias + '/', timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
        return

    # Create a new device_configuration if it doesn't already exist!
    # Note: session is ignored
    def add_device_config(self, cfg, session=None):
        try:
            xx = requests.get(self.prefix + 'add_device_config/' + cfg + '/',
                              timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return

        if not xx['success']:
            logging.error('%s' % xx['msg'])
        return

    # Return a list of all device configurations.
    def get_device_configs(self):
        try:
            xx = requests.get(self.prefix + 'get_device_configs/').json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
        return xx['value']

    # Return a list of all devices in an alias/hutch.
    def get_devices(self, alias, hutch=None):
        if hutch is None:
            hutch = self.hutch
        try:
            xx = requests.get(self.prefix + 'get_devices/' + hutch + '/' +
                              alias + '/', timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
        return xx['value']

    # Modify the current configuration for a specific device, adding it if
    # necessary.  name is the device and value is a json dictionary for the
    # configuration.  Return the new configuration key if successful and
    # raise an error if we fail.
    def modify_device(self, alias, value, hutch=None):
        if hutch is None:
            hutch = self.hutch

        alist = self.get_aliases(hutch)
        if not alias in alist:
            raise NameError("modify_device: %s is not a configuration name!"
                            % alias)
        if isinstance(value, cdict):
            value = value.typed_json()
        if not isinstance(value, dict):
            raise TypeError("modify_device: value is not a dictionary!")
        if not "detType:RO" in value.keys():
            raise ValueError("modify_device: value has no detType set!")
        if not "detName:RO" in value.keys():
            raise ValueError("modify_device: value has no detName set!")

        try:
            xx = requests.get(self.prefix + 'modify_device/' + hutch + '/' +
                              alias + '/', timeout=self.timeout,
                              json=value).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            raise
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])

        return xx['value']

    # Print all of the device configurations, or all of the configurations
    # for a specified device.
    def print_device_configs(self, name="device_configurations"):
        try:
            xx = requests.get(self.prefix + 'print_device_configs/' + name + '/',
                              timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return

        if not xx['success']:
            logging.error('%s' % xx['msg'])
        else:
            print(xx['value'].strip())

    # Print all of the configurations for the hutch.
    def print_configs(self, hutch=None):
        if hutch is None:
            hutch = self.hutch
        try:
            xx = requests.get(self.prefix + 'print_configs/' + hutch + '/',
                              timeout=self.timeout).json()
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return

        if not xx['success']:
            logging.error('%s' % xx['msg'])
        else:
            print(xx['value'].strip())

    # Transfer a configuration from another hutch to the current hutch,
    # returning the new key.
    # On error return zero.
    def transfer_config(self, oldhutch, oldalias, olddevice, newalias,
                        newdevice):
        try:
            # read configuration from old location
            read_val = self.get_configuration(oldalias, olddevice, hutch=oldhutch)

            # check for errors
            if not read_val:
                logging.error('get_configuration returned empty eonfig.')
                return 0

            # set detName
            read_val['detName:RO'] = newdevice 

            # write configuration to new location
            write_val = self.modify_device(newalias, read_val, hutch=self.hutch)
        except Exception as ex:
            logging.error('%s' % ex)
            return 0

        return write_val

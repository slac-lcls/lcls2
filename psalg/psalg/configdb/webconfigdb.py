import requests
from .typed_json import cdict

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

        if create:
            requests.get(self.prefix + 'create_collections/' + hutch + '/')

    # Retrieve the configuration of the device with the specified alias.
    # This returns a dictionary where the keys are the collection names and the 
    # values are typed JSON objects representing the device configuration(s).
    def get_configuration(self, alias, device, hutch=None):
        if hutch is None:
            hutch = self.hutch
        xx = requests.get(self.prefix + 'get_configuration/' + hutch + '/' +
                          alias + '/' + device + '/').json()
        return xx

    # Return a list of all hutches.
    def get_hutches(self):
        xx = requests.get(self.prefix + 'get_hutches/').json()
        return xx

    # Return a list of all aliases in the hutch.
    def get_aliases(self, hutch=None):
        if hutch is None:
            hutch = self.hutch
        xx = requests.get(self.prefix + 'get_aliases/' + hutch + '/').json()
        return xx

    # Create a new alias in the hutch, if it doesn't already exist.
    def add_alias(self, alias):
        xx = requests.get(self.prefix + 'add_alias/' + self.hutch + '/' +
                          alias + '/').json()
        return xx

    # Create a new device_configuration if it doesn't already exist!
    # Note: session is ignored
    def add_device_config(self, cfg, session=None):
        xx = requests.get(self.prefix + 'add_device_config/' + cfg + '/').json()
        return xx

    # Return a list of all device configurations.
    def get_device_configs(self):
        xx = requests.get(self.prefix + 'get_device_configs/').json()
        return xx

    # Return a list of all devices in an alias/hutch.
    def get_devices(self, alias, hutch=None):
        if hutch is None:
            hutch = self.hutch
        xx = requests.get(self.prefix + 'get_devices/' + hutch + '/' +
                          alias + '/').json()
        return xx

    # Modify the current configuration for a specific device, adding it if
    # necessary.  name is the device and value is a json dictionary for the
    # configuration.  Return the new configuration key if successful and
    # raise an error if we fail.
    def modify_device(self, alias, value, hutch=None):
        if hutch is None:
            hutch = self.hutch

        if isinstance(value, cdict):
            value = value.typed_json()
        if not isinstance(value, dict):
            raise TypeError("modify_device: value is not a dictionary!")
        if not "detType:RO" in value.keys():
            raise ValueError("modify_device: value has no detType set!")
        if not "detName:RO" in value.keys():
            raise ValueError("modify_device: value has no detName set!")

        xx = requests.get(self.prefix + 'modify_device/' + hutch + '/' +
                          alias + '/', json=value).json()
        return xx

    # Print all of the device configurations, or all of the configurations
    # for a specified device.
    def print_device_configs(self, name="device_configurations"):
        xx = requests.get(self.prefix + 'print_device_configs/' + name + '/').json()
        print(xx.strip())

    # Print all of the configurations for the hutch.
    def print_configs(self, hutch=None):
        if hutch is None:
            hutch = self.hutch
        xx = requests.get(self.prefix + 'print_configs/' + hutch + '/').json()
        print(xx.strip())

    # Transfer a configuration from another hutch to the current hutch,
    # returning the new key.
    def transfer_config(self, oldhutch, oldalias, olddevice, newalias,
                        newdevice):
        # read configuration from old location
        value = self.get_configuration(oldalias, olddevice, hutch=oldhutch)

        # set detName
        value['detName:RO'] = newdevice 

        # write configuration to new location
        key = self.modify_device(newalias, value, hutch=self.hutch)

        return key

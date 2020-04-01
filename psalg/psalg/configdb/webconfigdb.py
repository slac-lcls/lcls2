import requests

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
        self.create = create
        self.prefix = url + '/' + root + '/'

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

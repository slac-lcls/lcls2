import requests
from requests.auth import HTTPBasicAuth
from krtc import KerberosTicket
from urllib.parse import urlparse
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
    #     url    - e.g. "https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws"
    #     hutch  - Instrument name, e.g. "tmo"
    #     create - If True, try to create the database and collections
    #              for the hutch, device configurations, and counters.
    #     root   - Database name, usually "configDB"
    #     user   - User for HTTP authentication
    #     password - Password for HTTP authentication
    def __init__(self, url, hutch, create=False, root="NONE", user="xppopr", password="pcds"):
        if root == "NONE":
            raise Exception("configdb: Must specify root!")
        self.hutch  = hutch
        self.prefix = url.strip('/') + '/' + root + '/'
        self.host = urlparse(self.prefix).hostname
        self.timeout = 3.05     # timeout for http requests
        self.user = user
        self.password = password

        if create:
            try:
                xx = self._get_response('create_collections/' + hutch + '/')
            except requests.exceptions.RequestException as ex:
                logging.error('Web server error: %s' % ex)
            except Exception as ex:
                logging.error('%s' % ex)
            else:
                if not xx['success']:
                    logging.error('%s' % xx['msg'])

    # Return json response.
    # Raise exception on error.
    def _get_response(self, cmd, *, json=None):
        if 'ws-auth' in self.prefix:
            # basic authentication
            resp = requests.get(self.prefix + cmd,
                                auth=HTTPBasicAuth(self.user, self.password),
                                json=json,
                                timeout=self.timeout)
        elif 'ws-kerb' in self.prefix:
            # kerberos authentication
            resp = requests.get(self.prefix + cmd,
                                **{"headers": KerberosTicket('HTTP@' + self.host).getAuthHeaders()},
                                json=json,
                                timeout=self.timeout)
        else:
            # no authentication
            resp = requests.get(self.prefix + cmd,
                                json=json,
                                timeout=self.timeout)
        # raise exception if status is not ok
        resp.raise_for_status()
        return resp.json()

    # Retrieve the configuration of the device with the specified alias.
    # This returns a dictionary where the keys are the collection names and the 
    # values are typed JSON objects representing the device configuration(s).
    # On error return an empty dictionary.
    def get_configuration(self, alias, device, hutch=None):
        if hutch is None:
            hutch = self.hutch
        try:
            xx = self._get_response('get_configuration/' + hutch + '/' +
                                    alias + '/' + device + '/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return dict()
        except Exception as ex:
            logging.error('%s' % ex)
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
            xx = self._get_response('get_history/' + hutch + '/' +
                                    alias + '/' + device + '/',
                                    json=value)
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            xx = []
        except Exception as ex:
            logging.error('%s' % ex)
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
            xx = self._get_response('get_version/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return dict()
        except Exception as ex:
            logging.error('%s' % ex)
            return dict()
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
                return dict()
        return xx['value']

    # Return the highest key for the specified alias, or highest + 1 for all
    # aliases in the hutch if not specified.
    # On error return an empty list.
    def get_key(self, alias=None, hutch=None, session=None):
        if hutch is None:
            hutch = self.hutch
        try:
            if alias is None:
                xx = self._get_response('get_key/' + hutch + '/')
            else:
                xx = self._get_response('get_key/' + hutch + '/?alias=%s' % alias)
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        except Exception as ex:
            logging.error('%s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
        return xx['value']

    # Return a list of all hutches available in the config db.
    # On error return an empty list.
    def get_hutches(self):
        try:
            xx = self._get_response('get_hutches/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        except Exception as ex:
            logging.error('%s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
                return []
        return xx['value']

    # Return a list of all aliases in the hutch.
    # On error return an empty list.
    def get_aliases(self, hutch=None):
        if hutch is None:
            hutch = self.hutch
        try:
            xx = self._get_response('get_aliases/' + hutch + '/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        except Exception as ex:
            logging.error('%s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
                return []
        return xx['value']

    # Create a new alias in the hutch, if it doesn't already exist.
    def add_alias(self, alias):
        try:
            xx = self._get_response('add_alias/' + self.hutch + '/' + alias + '/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
        except Exception as ex:
            logging.error('%s' % ex)
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
        return

    # Create a new device_configuration if it doesn't already exist!
    # Note: session is ignored
    def add_device_config(self, cfg, session=None):
        if cfg in self.get_device_configs():
            # already exists!
            logging.info('device configuration \'%s\' already exists' % cfg)
            return
        try:
            xx = self._get_response('add_device_config/' + cfg + '/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return
        except Exception as ex:
            logging.error('%s' % ex)
            return

        if not xx['success']:
            logging.error('%s' % xx['msg'])
        return

    # Return a list of all device configurations.
    def get_device_configs(self):
        try:
            xx = self._get_response('get_device_configs/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        except Exception as ex:
            logging.error('%s' % ex)
            return []
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
                return []
        return xx['value']

    # Return a list of all devices in an alias/hutch.
    def get_devices(self, alias, hutch=None):
        if hutch is None:
            hutch = self.hutch
        try:
            xx = self._get_response('get_devices/' + hutch + '/' + alias + '/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return []
        except Exception as ex:
            logging.error('%s' % ex)
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
            xx = self._get_response('modify_device/' + hutch + '/' + alias + '/',
                                    json=value)
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            raise
        else:
            if not xx['success']:
                logging.error('%s' % xx['msg'])
                raise Exception("modify_device: operation failed!")

        return xx['value']

    # Print all of the device configurations, or all of the configurations
    # for a specified device.
    def print_device_configs(self, name="device_configurations"):
        try:
            xx = self._get_response('print_device_configs/' + name + '/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return
        except Exception as ex:
            logging.error('%s' % ex)
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
            xx = self._get_response('print_configs/' + hutch + '/')
        except requests.exceptions.RequestException as ex:
            logging.error('Web server error: %s' % ex)
            return
        except Exception as ex:
            logging.error('%s' % ex)
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
                logging.error('get_configuration returned empty config.')
                return 0

            # set detName
            read_val['detName:RO'] = newdevice 

            # write configuration to new location
            write_val = self.modify_device(newalias, read_val, hutch=self.hutch)
        except Exception as ex:
            logging.error('%s' % ex)
            return 0

        return write_val

# ------------------------------------------------------------------------------
# configdb CLI
# ------------------------------------------------------------------------------

import sys
import argparse
import pprint

# Parse a device name into 4 elements.
# Input format: <hutch>/<alias>/<device>_<segment>
# Returns: hutch, alias, device, segment
# On error raises NameError.
def _parse_device4(name):
    error_txt = 'Name \'%s\' does not match <hutch>/<alias>/<device>_<segment>' % name
    try:
        split1 = name.rsplit('_', maxsplit=1)
        segment = int(split1[1])
    except Exception:
        raise NameError(error_txt)

    if len(split1) != 2:
        raise NameError(error_txt)

    split2 = split1[0].split('/')
    if len(split2) != 3:
        raise NameError(error_txt)

    return (split2[0], split2[1], split2[2], segment)

def _cat(args):
    try:
        hutch, alias, dev, seg = _parse_device4(args.src)
    except NameError as ex:
        print('%s' % ex) 
        sys.exit(1)

    # authentication is not required, adjust url accordingly
    url = args.url.replace('ws-auth', 'ws').replace('ws-kerb', 'ws')

    # get configuration and pretty print it
    mycdb = configdb(url, hutch, root=args.root)
    xx = mycdb.get_configuration(alias, '%s_%d' % (dev, seg), hutch)
    if len(xx) > 0:
        pprint.pprint(xx)

def _cp(args):
    try:
        oldhutch, oldalias, olddev, oldseg = _parse_device4(args.src)
        newhutch, newalias, newdev, newseg = _parse_device4(args.dst)
    except NameError as ex:
        print('%s' % ex) 
        sys.exit(1)

    # transfer configuration
    mycdb = configdb(args.url, newhutch, create=args.create, root=args.root,
                     user=args.user, password=args.password)
    if args.create:
        mycdb.add_alias(newalias)
    retval = mycdb.transfer_config(oldhutch, oldalias, '%s_%d' % (olddev, oldseg),
                                   newalias, '%s_%d' % (newdev, newseg))
    if retval == 0:
        print('failed to transfer configuration')
        sys.exit(1)

def _ls(args):
    # authentication is not required, adjust url accordingly
    url = args.url.replace('ws-auth', 'ws').replace('ws-kerb', 'ws')
    mycdb = configdb(url, None, root=args.root)

    if args.src is None:
        src_list = []
    else:
        src_list = args.src.split('/')

    if len(src_list) == 0:
        # get list of hutches and print them
        for hutch in mycdb.get_hutches():
            print(hutch)
    elif len(src_list) == 1:
        # get list of aliases in hutch and print them
        for alias in mycdb.get_aliases(src_list[0]):
            print(alias)
    elif len(src_list) == 2:
        # get list of devices in hutch/alias and print them
        for device in mycdb.get_devices(src_list[1], src_list[0]):
            print(device)
    else:
        print('Name \'%s\' does not match <hutch>[/<alias>]' % args.src)
        sys.exit(1)

def main():

    # create the top-level parser
    parser = argparse.ArgumentParser(description='configuration database CLI')
    parser.add_argument('--url', default='https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/',
                        help='configuration database connection')
    parser.add_argument('--root', default='configDB', help='configuration database root (default: configDB)')
    subparsers = parser.add_subparsers()

    # create the parser for the "cat" command
    parser_cat = subparsers.add_parser('cat', help='print a configuration')
    parser_cat.add_argument('src', help='source: <hutch>/<alias>/<device>_<segment>')
    parser_cat.set_defaults(func=_cat)

    # create the parser for the "cp" command
    parser_cp = subparsers.add_parser('cp', help='copy a configuration')
    parser_cp.add_argument('src', help='source: <hutch>/<alias>/<device>_<segment>')
    parser_cp.add_argument('dst', help='destination: <hutch>/<alias>/<device>_<segment>')
    parser_cp.add_argument('--user', default='xppopr', help='default: xppopr')
    parser_cp.add_argument('--password', default='pcds', help='default: pcds')
    parser_cp.add_argument('--create', action='store_true', help='create destination hutch or alias if needed')
    parser_cp.set_defaults(func=_cp)

    # create the parser for the "ls" command
    parser_ls = subparsers.add_parser('ls', help='list directory contents')
    parser_ls.add_argument('src', help='source: <hutch>[/<alias>]', nargs='?', default=None)
    parser_ls.set_defaults(func=_ls)

    # parse the args and call whatever function was selected
    args = parser.parse_args()
    try:
        subcommand = args.func
    except Exception:
        parser.print_help(sys.stderr)
        sys.exit(1)
    subcommand(args)

if __name__ == '__main__':
    main()

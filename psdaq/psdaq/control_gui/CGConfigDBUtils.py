#--------------------
"""
:py:class:`CGConfigDBUtils` - import utilities from configdb using singleton
============================================================================

Usage::

    # Import and make configdb object with non-default parameters:
    from psdaq.control_gui.CGConfigDBUtils import get_configdb, URI_CONFIGDB, ROOT_CONFIGDB
    confdb = get_configdb(uri=URI_CONFIGDB, hutch=INSTRUMENT, create=False, root=ROOT_CONFIGDB, user="xppopr", password="pcds")

    # Methods - see :py:class:`CGWMainPartition`

See:
    - :py:class:`CGConfigDBUtils`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-03-25 by Mikhail Dubrovin
"""
#--------------------

#import logging
#logger = logging.getLogger(__name__)

URI_CONFIGDB = 'https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/'
ROOT_CONFIGDB = 'configDB'

#--------------------

from psdaq.configdb.configdb import configdb

def get_configdb(uri=URI_CONFIGDB, hutch='tmo', create=False, root=ROOT_CONFIGDB, user='tmoopr', password='pcds') :
    return configdb(uri, hutch, create=create, root=root, user=user, password=password)

#--------------------

#confdb = get_configdb()

#--------------------

if __name__ == "__main__" :

    # Access the configuration database using:
    #from psdaq.configdb.configdb import configdb

    INSTRUMENT = 'TMO'

    confdb = get_configdb(URI_CONFIGDB, INSTRUMENT, create=False, root=ROOT_CONFIGDB, user='tmoopr', password='pcds')
              # <psdaq.configdb.configdb.configdb at 0x7fab765ebf98>

    # Find the hutches in the database:
    list_of_hutch_names = confdb.get_hutches() # ['TMO', 'CXI', etc.]
    print('list_of_hutch_names:', list_of_hutch_names)

    # Find the aliases in a hutch:
    list_of_alias_names = confdb.get_aliases(hutch=INSTRUMENT) # ['NOBEAM', 'BEAM']
    print('list_of_alias_names:', list_of_alias_names)

    # Find the device configurations available for all hutches:
    list_of_device_configuration_names = confdb.get_device_configs() #['test']
    print('list_of_device_configuration_names:', list_of_device_configuration_names)

    #Find the devices in an alias in a hutch:
    list_of_device_names_beam   = confdb.get_devices('BEAM', hutch=INSTRUMENT) # ['testdev0']
    print('list_of_device_names_beam:', list_of_device_names_beam)

    list_of_device_names_nobeam = confdb.get_devices('NOBEAM', hutch=INSTRUMENT) # ['testdev0']
    print('list_of_device_names_nobeam:', list_of_device_names_nobeam)

    #Retrieve a configuration for a device:
    config = confdb.get_configuration('BEAM', 'testdev0', hutch=INSTRUMENT)
    print('config:\n', config)
    #    The config is a dictionary: keys are device configuration names, and
    #    values are typed JSON dictionaries.

    #Modify a configuration for a device:
    #new_key = confdb.modify_device('BEAM', 'testdev0', config, hutch=INSTRUMENT)
    #    The config is a dictionary in the form retrieved from
    #    get_configuration, *not* a typed JSON dictionary!  If successful,
    #    this returns an integer key that can be used to refer to the new
    #    configuration of the alias. On failure, this routine raises an exception.
    #print('new_key:', new_key)

#--------------------

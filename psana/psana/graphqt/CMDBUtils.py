
#------------------------------
"""Class :py:class:`CMDBBUtils` utilities for calib manager DB methods
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMDBUtils.py

    # Import
    import psana.graphqt.CMDBUtils as dbu

    # See test at the EOF

See:
  - :class:`CMWMain`
  - :class:`CMWConfig`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2018-04-10 by Mikhail Dubrovin
"""
#------------------------------

import psana.pscalib.calib.MDBUtils as dbu

connect_to_server = dbu.connect_to_server
database_names    = dbu.database_names
database          = dbu.database
collection_names  = dbu.collection_names   
#db_prefixed_name  = dbu.db_prefixed_name   # ('') 
#delete_databases  = dbu.delete_databases   # (list_db_names)
#delete_collections= dbu.delete_collections # (dic_db_cols)
#collection_info   = dbu.collection_info    # (client, dbname, colname)

#------------------------------
from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger

_name = 'DCMDBUtils'

#------------------------------

def connect_client() :
    host = cp.cdb_host.value()
    port = cp.cdb_port.value()
    logger.debug('Connect client to host: %s port: %d' % (host, port), _name)
    return dbu.connect_to_server(host, port)

#------------------------------

def delete_databases(list_db_names) :
    """Delete databases specified in the list_db_names
    """
    client = connect_client()
    logger.debug('Delete databases:\n  %s' % ('\n  '.join(list_db_names)), _name)
    dbu.delete_databases(client, list_db_names)

#------------------------------

def delete_collections(dic_db_cols) :
    """Delete collections specified in the dic_db_cols consisting of pairs {dbname:lstcols}
    """
    msg = 'Delete collections:'
    client = connect_client()
    for dbname, lstcols in dic_db_cols.items() :
        db = dbu.database(client, dbname)
        msg += '\nFrom database: %s delete collections:\n  %s' % (dbname, '\n  '.join(lstcols))
        dbu.delete_collections(db, lstcols)
    logger.debug(msg, _name)

#------------------------------

def collection_info(dbname, colname) :
    """Delete collections specified in the dic_db_cols consisting of pairs {dbname:lstcols}
    """
    msg = 'Delete collections:'
    client = connect_client()
    return dbu.collection_info(client, dbname, colname)

#------------------------------



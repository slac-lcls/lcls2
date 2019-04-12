
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
import logging
logger = logging.getLogger(__name__)
#_name = 'DCMDBUtils'
#from psana.pyalgos.generic.Logger import logger

import psana.pscalib.calib.MDBUtils as dbu

ObjectId          = dbu.ObjectId

connect_to_server = dbu.connect_to_server
database_names    = dbu.database_names
database          = dbu.database
collection_names  = dbu.collection_names   
collection        = dbu.collection
timestamp_id      = dbu.timestamp_id
doc_add_id_ts     = dbu.doc_add_id_ts
db_prefixed_name  = dbu.db_prefixed_name
time_and_timestamp = dbu.time_and_timestamp
exportdb          = dbu.exportdb
importdb          = dbu.importdb
out_fname_prefix  = dbu.out_fname_prefix
save_doc_and_data_in_file = dbu.save_doc_and_data_in_file

#insert_data_and_doc = dbu.insert_data_and_doc
#document_info     = dbu.document_info
#db_prefixed_name  = dbu.db_prefixed_name   # ('') 
#delete_databases  = dbu.delete_databases   # (list_db_names)
#delete_collections= dbu.delete_collections # (dic_db_cols)
#collection_info   = dbu.collection_info    # (client, dbname, colname)

#------------------------------
from psana.graphqt.CMConfigParameters import cp

#------------------------------

def connect_client(host=None, port=None, user=cp.user, upwd=cp.upwd) : # user=dbu.cc.USERNAME
    _host = cp.cdb_host.value() if host is None else host
    _port = cp.cdb_port.value() if port is None else port
    #logger.debug('CMDBBUtils: Connect client to host: %s port: %d user: %s upwd: %s' % (_host, _port, user, upwd))
    return dbu.connect_to_server(_host, _port, user, upwd)
           # if cp.upwd else dbu.connect_to_server(_host, _port, cp.user)

#------------------------------

def delete_databases(list_db_names) :
    """Delete databases specified in the list_db_names
    """
    client = connect_client()
    logger.debug('Delete databases:\n  %s' % ('\n  '.join(list_db_names)))
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
    logger.debug(msg)

#------------------------------

def delete_documents(dbname, colname, doc_ids) :
    """Delete documents with _id-s in doc_ids from dbname, colname
    """
    #logger.debug('Deleting documents:\n  %s' % ('\n  '.join(doc_ids)))
    client = connect_client()
    db, fs = dbu.db_and_fs(client, dbname)
    col = collection(db, colname)
    #msg = 'Deleted documents from db: %s col: %s' % (dbname, colname)
    for s in doc_ids :
        oid = ObjectId(s)
        doc = dbu.find_doc(col, query={'_id':oid})
        if doc is None : continue
        #msg += '\n  %s and its data' % doc.get('_id', 'N/A')
        dbu.del_document_data(doc, fs)
        dbu.delete_document_from_collection(col, oid)

    #logger.debug(msg)

#------------------------------
#------------------------------

def insert_document_and_data(dbname, colname, doc, data) :
    client = connect_client()
    db, fs = dbu.db_and_fs(client, dbname)
    col = collection(db, colname)
    id_data, id_doc = dbu.insert_data_and_doc(data, fs, col, **doc)
    return id_data, id_doc

#------------------------------

def get_data_for_doc(dbname, doc) :
    client = connect_client()
    db, fs = dbu.db_and_fs(client, dbname)
    return dbu.get_data_for_doc(fs, doc)

#------------------------------
#------------------------------

def collection_info(dbname, colname) :
    """Delete collections specified in the dic_db_cols consisting of pairs {dbname:lstcols}
    """
    msg = 'Delete collections:'
    client = connect_client()
    return dbu.collection_info(client, dbname, colname)

#------------------------------

def list_of_documents(dbname, colname) :
    client = connect_client()
    db = database(client, dbname)
    #db, fs = dbu.db_and_fs(client, dbname='cdb-cxi12345')
    col = collection(db, colname)
    docs = col.find().sort('_id', dbu.DESCENDING)
    return [d for d in docs]

#------------------------------

def document_info(doc, keys=('time_sec','time_stamp','experiment',\
                  'detector','ctype','run','id_data_ts','data_type','data_dtype', '_id'),\
                  fmt='%10s %24s %11s %24s %16s %4s %30s %10s %10s %24s') :
    """The same as dbu.document_info, but with different default parameters (added _id).
    """
    return dbu.document_info(doc, keys, fmt)

#------------------------------


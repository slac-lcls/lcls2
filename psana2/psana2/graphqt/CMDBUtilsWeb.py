import logging
logger = logging.getLogger(__name__)
#_name = 'DCMDBUtilsWeb'
#from psana2.pyalgos.generic.Logger import logger

import psana2.pscalib.calib.MDBWebUtils as wu

database_names   = wu.database_names   # () -> list of str
collection_names = wu.collection_names # (dbname) -> list of str
collection_info  = wu.collection_info  # (dbname, colname) -> str
document_info    = wu.mu.document_info # (doc) -> str
document_keys    = wu.mu.document_keys # (doc) -> str
list_of_documents= wu.list_of_documents
doc_add_id_ts    = wu.mu.doc_add_id_ts
ObjectId         = wu.mu.ObjectId
get_data_for_doc = wu.get_data_for_doc
timestamp_id     = wu.mu.timestamp_id
delete_documents = wu.delete_documents
delete_collections = wu.delete_collections
delete_databases = wu.delete_databases
time_and_timestamp = wu.mu.time_and_timestamp
time_and_timestamp = wu.mu.time_and_timestamp
out_fname_prefix = wu.mu.out_fname_prefix
save_doc_and_data_in_file = wu.mu.save_doc_and_data_in_file
db_prefixed_name = wu.mu.db_prefixed_name
insert_document_and_data = wu.insert_document_and_data

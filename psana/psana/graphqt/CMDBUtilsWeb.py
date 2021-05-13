import logging
logger = logging.getLogger(__name__)
#_name = 'DCMDBUtilsWeb'
#from psana.pyalgos.generic.Logger import logger

import psana.pscalib.calib.MDBWebUtils as wu

database_names   = wu.database_names   # () -> list of str
collection_names = wu.collection_names # (dbname) -> list of str
collection_info  = wu.collection_info  # (dbname, colname) -> str
document_info    = wu.mu.document_info # (doc) -> str
document_keys    = wu.mu.document_keys # (doc) -> str

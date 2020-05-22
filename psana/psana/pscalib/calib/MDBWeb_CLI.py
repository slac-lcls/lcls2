#------------------------------
"""
Created on 2020-05-15 by Mikhail Dubrovin
"""
#------------------------------

from psana.pscalib.calib.MDB_CLI import * # gu, mu, etc
import psana.pscalib.calib.MDBWebUtils as wu
cc = wu.cc
logger = logging.getLogger(__name__)

#------------------------------

class MDBWeb_CLI(MDB_CLI):

    def __init__(self, parser): 
        MDB_CLI.__init__(self, parser)


    def _warning(self): logger.warning('MDBWeb_CLI: TBD for mode: %s' % self.mode)


    def print_content(self):
        logger.info(wu.info_webclient(**self.kwargs))


    def deldoc(self):
        kwa = self.kwargs
        dbname = mu.get_dbname(**kwa)
        colname = mu.get_colname(**kwa)
        docid = kwa.get('docid', None)
        if None in (dbname, colname, docid):
            logger.warning('CAN NOT DELETE DOCUMENT for DB/collection/docid: %s/%s/%s\n%s'%\
                           (str(dbname), str(colname), str(docid),\
                            wu.info_docs(dbname, colname, query={}, url=cc.URL, strlen=150)))
            return
                
        docs = wu.find_docs(dbname, colname, query={}, url=cc.URL)
        logger.info('deldoc DB/collection/docid: %s/%s/%s from list of docs:%s' % (str(dbname), str(colname), str(docid),\
                    wu.info_docs_list(docs, strlen=150)))
        
        if kwa.get('confirm', False):
            resp = wu.delete_document_and_data(dbname, colname, docid, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
            logger.info('deldoc for DB/collection/docid: %s/%s/%s resp=%s' % (str(dbname), str(colname), str(docid), str(resp)))
        else:
            mu.request_confirmation()


    def delcol(self):
        kwa = self.kwargs
        dbname = mu.get_dbname(**kwa)
        colname = mu.get_colname(**kwa)
        if None in (dbname, colname):
            logger.warning('CAN NOT DELETE COLLECTION for DB/collection: %s/%s' % (str(dbname), str(colname)))
 
        confirm = kwa.get('confirm', False)
        # delete GridFS data associated with collection documents
        docs = wu.find_docs(dbname, colname, query={}, url=cc.URL)

        if docs is None:
            logger.warning('DB/collection %s/%s DOCUMENTS NOT FOUND' % (dbname, colname))
        else:
            s = 'delcol DB/collection %s/%s contains %d documents:%s' % (dbname, colname, len(docs),\
                 wu.info_docs_list(docs, strlen=150))
            logger.info(s)

        if confirm:
            for doc in docs:
                data_id = doc.get('id_data', None)
                resp = wu.delete_data(dbname, data_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
        else:
            mu.request_confirmation()
            return

        # delete collection itself
        logger.info('before delcol %s/%s collections: %s' % (dbname, colname, str(wu.collection_names(dbname, url=cc.URL))))
        if confirm:
            resp = wu.delete_collection(dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
            logger.info('after delcol %s/%s collections: %s' % (dbname, colname, str(wu.collection_names(dbname, url=cc.URL))))
        else:
            mu.request_confirmation()


    def deldb(self):
        dbname = mu.get_dbname(**self.kwargs)

        logger.info('MDBWeb_CLI mode "%s" database "%s"' % (self.mode, dbname))

        ptrn = mu.db_prefixed_name('') if self.kwargs.get('cdbonly', True) else None
        dbnames = wu.database_names(pattern=ptrn)
        if not (dbname in dbnames):
            logger.warning('Database %s is not found in the list of known:\n%s' % (dbname, wu.str_formatted_list(dbnames)))
            return

        logger.debug('Databases before %s:\n%s' % (self.mode, wu.str_formatted_list(dbnames)))
        if self.kwargs.get('confirm', False) :
            resp = wu.delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
            dbnames = wu.database_names(pattern=ptrn)
            logger.debug('Databases after %s:\n%s' % (self.mode, wu.str_formatted_list(dbnames)))
        else :
            mu.request_confirmation()

    def get(self):
        # self._warning()
        kwa        = self.kwargs
        det        = kwa.get('detector', None)
        exp        = kwa.get('experiment', None)
        ctype      = kwa.get('ctype', None)
        run        = kwa.get('run', None)
        vers       = kwa.get('version', None)
        prefix     = kwa.get('iofname', None)
        time_sec   = kwa.get('time_sec', None)
        time_stamp = kwa.get('time_stamp', None)
        tsformat   = kwa.get('tsformat', '%Y-%m-%dT%H:%M:%S%z') # e.g. 2020-05-22T10:39:07-0700
        if time_stamp is not None:
           time_sec = gu.time_sec_from_stamp(tsformat, time_stamp)

        data,doc = wu.calib_constants(det, exp, ctype, run, time_sec, vers, url=cc.URL)
        logger.info('data: %s' % str(data)[:150])
        logger.info('doc: %s' % str(doc))

        if prefix is None : prefix = mu.out_fname_prefix(**doc)
        mu.save_doc_and_data_in_file(doc, data, prefix, control={'data' : True, 'meta' : True})


    #def add(self): self._warning()


    def test(self):
        self._warning()
        logger.warning('MDBWeb_CLI.test')


    def dispatcher(self):
        mode = self.mode
        logger.debug('mode: %s' % mode)
        # Reimplemented web access methods
        if   mode == 'print' : self.print_content()
        elif mode == 'deldoc': self.deldoc()
        elif mode == 'delcol': self.delcol()
        elif mode == 'deldb' : self.deldb()
        elif mode == 'get'   : self.get()

        # pymongo access from MDB_CLI
        elif mode == 'add'   : self.add()

        elif mode == 'convert': self.convert()
        elif mode == 'delall' : self.delall()
        elif mode == 'export' : self.exportdb()
        elif mode == 'import' : self.importdb()
        elif mode == 'test'   : self.test()

        else : logger.warning('Non-implemented command mode "%s"\n  Known modes: %s' % (mode,', '.join(MODES)))

#------------------------------

def cdb_web(parser):
    """Calibration Data Base Command Line Interface
    """
    MDBWeb_CLI(parser)

#------------------------------

if __name__ == "__main__":
    import sys
    sys.exit('Run command cdb -w ...')

#------------------------------

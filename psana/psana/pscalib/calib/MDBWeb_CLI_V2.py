
"""
Created on 2020-05-15 by Mikhail Dubrovin
"""

import os
import sys
from psana.pscalib.calib.MDB_CLI import * # gu, mu, etc
import psana.pscalib.calib.MDBWebUtils as wu
import psana.detector.utils_psana as up
from psana.detector.Utils import info_dict
import logging
logger = logging.getLogger(__name__)
cc = wu.cc


def datasource_kwargs_from_string(str_dskwargs, detname=None):
    dskwargs = up.datasource_kwargs_from_string(str_dskwargs, detname=detname)
    logger.info('DataSource kwargs: %s' % str(dskwargs))
    return dskwargs


class MDBWeb_CLI(MDB_CLI):

    def __init__(self, parser):
        MDB_CLI.__init__(self, parser)

    def _warning(self): logger.warning('MDBWeb_CLI: TBD for mode: %s' % self.mode)

    def print_content(self):
        logger.info(wu.info_webclient(**self.kwargs))

    def deldoc(self):
        """Deletes document and associated data for specified docid. DB and collection names are evaluated from input parameters."""
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
        """Deletes collection and all associated data. DB and collection names are evaluated from input parameters."""
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
        """Deletes data base. DB name is evaluated from input parameters."""
        dbname = mu.get_dbname(**self.kwargs)

        logger.info('MDBWeb_CLI mode "%s" database "%s"' % (self.mode, dbname))

        ptrn = mu.db_prefixed_name('') if self.kwargs.get('cdbonly', True) else None
        dbnames = wu.database_names(pattern=ptrn)
        if not (dbname in dbnames):
            logger.warning('Database %s is not found in the list of known:\n%s' % (dbname, wu.str_formatted_list(dbnames)))
            return

        logger.debug('Databases before %s:\n%s' % (self.mode, wu.str_formatted_list(dbnames)))
        if self.kwargs.get('confirm', False):
            resp = wu.delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
            dbnames = wu.database_names(pattern=ptrn)
            logger.debug('Databases after %s:\n%s' % (self.mode, wu.str_formatted_list(dbnames)))
        else:
            mu.request_confirmation()


    def get(self):
        """Finds requested document and associated data and saves them in files."""
        # self._warning()
        kwa        = self.kwargs
        _dskwargs  = kwa.get('dskwargs', None)
        dskwargs = datasource_kwargs_from_string(_dskwargs, detname=None)
        exp = dskwargs['exp']
        run = dskwargs['run']
        kwa['experiment'] = exp
        kwa['run'] = run
        det        = kwa.get('detector', None)
        ctype      = kwa.get('ctype', None)
        vers       = kwa.get('version', None)
        prefix     = kwa.get('iofname', None)
        dbsuffix   = kwa.get('dbsuffix', '')
        time_sec   = kwa.get('time_sec', None)
        time_stamp = kwa.get('time_stamp', None)
        tsformat   = kwa.get('tsformat', '%Y-%m-%dT%H:%M:%S%z') # e.g. 2020-05-22T10:39:07-0700
        if time_stamp is not None:
           time_sec = gu.time_sec_from_stamp(tsformat, time_stamp)

        print('XXXX exp: %s run: %d det: %s ctype: %s time_sec: %d vers: %s dbsuffix: %s' % (exp, run, det, ctype, time_sec, vers, dbsuffix))
        #self.defs = vars(parser.parse_args([]))
        print('XXXX defs: %s' % self.defs)
        _vers = None if vers == self.defs['version'] else vers
        print('XXXX _vers: %s' % _vers)
        resp = wu.calib_constants(det, exp, ctype, run, time_sec, vers=_vers, url=cc.URL, dbsuffix=dbsuffix)
        if resp is None:
            sys.exit('wu.calib_constants returns None')
        data, doc = resp
        logger.debug('data: %s' % str(data)[:150])
        logger.info('doc: %s' % str(doc))

        if doc is None or data is None:
            logger.warning('files for doc and data are not saved')
            return

        if prefix is None: prefix = mu.out_fname_prefix(**doc)
        mu.save_doc_and_data_in_file(doc, data, prefix, control={'data': True, 'meta': True})

    def add(self):
        """Adds calibration constants to database from file."""
        kwa = self.kwargs
        print('input kwa: %s' % str(kwa))
        _dskwargs = kwa.get('dskwargs', None)
        run_beg   = kwa.get('run_beg', 'None')
        fname     = kwa.get('iofname', 'None')
        ctype     = kwa.get('ctype', 'None')
        dtype     = kwa.get('dtype', 'None')
        det       = kwa.get('detector', None)
        dbsuffix  = kwa.get('dbsuffix', None)
        verb      = self.strloglev == 'DEBUG'
        dskwargs  = datasource_kwargs_from_string(_dskwargs, detname=None)
        exp = dskwargs['exp']
        run = dskwargs['run']
        kwa['run'] = run if run_beg is None else run_beg
        kwa['run_orig'] = run
        kwa['experiment'] = exp = dskwargs['exp']

        d = up.get_config_info_for_dataset_detname(**kwa)

        print('\nmetadata from DataSource run and run.Detector:', d)
        kwa['shortname'] = shortname = d.get('shortname', None)
        kwa['longname'] = longname = d.get('longname', None)
        kwa['detname'] = det
        kwa['dettype'] = d['dettype']
        kwa['det'] = det = shortname
        kwa['tstamp_orig'] = d['tstamp_orig']
        kwa['trun_sec']    = d['trun_sec']
        kwa['tsec_orig']   = d['tsec_orig']
        kwa['run_orig']    = d['run_orig']

        if not os.path.exists(fname):
            sys.exit('\nNot found file: %s' % fname)
        data = mu.data_from_file(fname, ctype, dtype, verb)

        s = info_dict(kwa, fmt='  %12s: %s', sep='\n')
        logger.info('\nMetadata for document in DB:\n%s\n' % s)
        resp = wu.deploy_constants(data, exp, longname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwa)
        #id_data_exp, id_data_det, id_doc_exp, id_doc_det = resp if resp is not None

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
        elif mode == 'add'   : self.add()

        # pymongo access from MDB_CLI
#        elif mode == 'convert': self.convert()
#        elif mode == 'delall' : self.delall()
#        elif mode == 'export' : self.exportdb()
#        elif mode == 'import' : self.importdb()
#        elif mode == 'test'   : self.test()

        else: logger.warning('Non-implemented command mode "%s"\n  Known modes: %s' % (mode,', '.join(MODES)))

def cdb_web(parser):
    """Calibration Data Base Command Line Interface"""
    MDBWeb_CLI(parser)

if __name__ == "__main__":
    import sys
    sys.exit('Run command cdb -w ...')

# EOF

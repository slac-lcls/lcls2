#------------------------------
"""
Created on 2018-02-23 by Mikhail Dubrovin
"""
#------------------------------

import os
from psana.pyalgos.generic.Utils import print_kwargs, print_parser, is_in_command_line

import psana.pyalgos.generic.Utils as gu
from psana.pyalgos.generic.NDArrUtils import info_ndarr #, print_ndarr
from psana.pscalib.calib.NDArrIO import load_txt, save_txt

import psana.pscalib.calib.MDBUtils as dbu # insert_constants, time_and_timestamp
import numpy as np

from psana.pyalgos.generic.logger import logging, config_logger
#import logging
logger = logging.getLogger(__name__)

MODES = ('print', 'convert', 'deldoc', 'delcol', 'deldb', 'delall', 'add', 'get', 'export', 'import', 'test')

#------------------------------

class MDB_CLI :

    def __init__(self, parser) : 
        self.unpack(parser)
        self.dispatcher()


    def unpack(self, parser) :
        """parser parameters:
          -  host      
          -  port      
          -  experiment
          -  detector  
          -  ctype     
          -  run       
          -  run_end       
          -  time_stamp
          -  time_sec
          -  version   
          -  iofname
          -  comment
          -  dbname
        """
        (popts, pargs) = parser.parse_args()
        #args = pargs
        #defs = vars(parser.get_default_values())

        self.mode = mode = pargs[0] if len(pargs) > 0 else 'print'

        kwargs = vars(popts)

        time_sec, time_stamp = dbu.time_and_timestamp(**kwargs)
        kwargs['time_sec']   = int(time_sec)
        kwargs['time_stamp'] = time_stamp
        kwargs['cli_mode']   = mode

        self.kwargs = kwargs
        self.defs = vars(parser.get_default_values())

        self.loglevel = kwargs.get('loglevel','DEBUG').upper()

        fmt = '%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'
        config_logger(loglevel=self.loglevel, fmt=fmt)

        if self.loglevel == 'DEBUG' :
            from psana.pyalgos.generic.Utils import print_kwargs, print_parser
            print(40*'_')
            print_parser(parser)
            print_kwargs(kwargs)
            fmt = '%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'


    def client(self) :
        kwargs = self.kwargs
        host  = kwargs.get('host',  None)
        port  = kwargs.get('port',  None)
        user  = kwargs.get('user',  None)
        upwd  = kwargs.get('upwd',  None)
        ctout = kwargs.get('ctout', 5000)
        stout = kwargs.get('stout', 30000)
        return dbu.connect_to_server(host, port, user, upwd, ctout, stout)


    def check_database(self, client, dbname) :
        if dbu.database_exists(client, dbname) : 
            return True
        logger.warning('Database "%s" is not available. See for deteals: cdb print'%(dbname))
        return False


    def print_content(self) :
        dbname = dbu.get_dbname(**self.kwargs)
        client = self.client()
        logger.info(dbu.client_info(client, level=2) if dbname is None else
                    dbu.database_info(client, dbname, level=3))


    def convert(self) :
        """Converts LCLS experiment calib directory to LCLS2 calibration database.
        """
        import psana.pscalib.calib.MDBConvertLCLS1 as cu
        kwargs = self.kwargs
        exp = kwargs.get('experiment', None)
        cu.scan_calib_for_experiment(exp, **kwargs)


    def delall(self) :
        """FOR DEVELOPMENT: Deletes all databases with prefix in the name.
        """
        mode, kwargs = self.mode, self.kwargs
        client = self.client()
        prefix = dbu.db_prefixed_name('')
        dbnames = dbu.database_names(client)
        logger.info('Databases before "%s":\n%s' % (mode, str(dbnames)))
        #confirm = kwargs.get('confirm', False)
        confirm = True
        for dbname in dbu.database_names(client) :
            if prefix in dbname :
                logger.info('delete %s' % dbname)
                if confirm : 
                    dbu.delete_database(client, dbname)
 
        if confirm :
            logger.info('Databases after "%s" %s:\n%s' % (mode, dbname, str(dbu.database_names(client))))
        else : 
            dbu.request_confirmation()


    def deldb(self) :
        """Deletes specified database.
        """
        mode, kwargs = self.mode, self.kwargs
        dbname = dbu.get_dbname(**kwargs)
        client = self.client()
        if not self.check_database(client, dbname) : return
        logger.info('Command mode "%s" database "%s"' % (mode, dbname))
        logger.info('Databases before:\n%s' % str(dbu.database_names(client)))

        if kwargs.get('confirm', False) :
            dbu.delete_database(client, dbname)
            logger.info('Databases after:\n%s' % str(dbu.database_names(client)))
        else :
            dbu.request_confirmation()


    def delcol(self) :
        """Deletes specified collection in the database.
        """
        mode, kwargs = self.mode, self.kwargs
        dbname  = dbu.get_dbname(**self.kwargs)
        client = self.client()
        if not self.check_database(client, dbname) : return

        detname = kwargs.get('detector', None)
        if detname is None :
            logger.warning('%s needs in the collection name. Please specify the detector name.'%(mode))
        colname = detname
        db, fs = dbu.db_and_fs(client, dbname)
        colnames = dbu.collection_names(db)
        logger.info('"%s" deletes collection "%s" from database "%s"' % (mode, colname, db.name))
        logger.info('Collections before "%s"' % str(colnames))
        logger.info(dbu.database_fs_info(db, gap=''))

        if not(colname in colnames) :
            logger.warning('db "%s" does not have collection "%s"' % (db.name, str(colname)))
            return

        if kwargs.get('confirm', False) : 
            col = dbu.collection(db, colname)
            dbu.del_collection_data(col, fs) # delete data in fs associated with collection col
            dbu.delete_collection_obj(col)
            logger.info('Collections after "%s"' % str(dbu.collection_names(db)))
            logger.info(dbu.database_fs_info(db, gap=''))
        else : 
            dbu.request_confirmation()


    def deldoc(self) :
        """Deletes specified document in the database.
        """
        mode, kwargs = self.mode, self.kwargs
        dbname  = dbu.get_dbname(**kwargs)
        client = self.client()
        if not self.check_database(client, dbname) : return

        detname = kwargs.get('detector', None)
        if detname is None :
            logger.warning('%s needs in the collection name. Please specify the detector name.'%(mode))
        colname = detname
        db, fs = dbu.db_and_fs(client, dbname)
        colnames = dbu.collection_names(db)

        if not(colname in colnames) : # dbu.collection_exists(db, colname)
            logger.warning('db "%s" does not have collection "%s"' % (db.name, str(colname)))
            return
            
        col = dbu.collection(db,colname)

        logger.info('command mode: "%s" db: "%s" collection: "%s"' % (mode, db.name, str(colname)))

        defs   = self.defs
        ctype  = kwargs.get('ctype',      None)
        run    = kwargs.get('run',        None)
        tsec   = kwargs.get('time_sec',   None)
        tstamp = kwargs.get('time_stamp', None)
        vers   = kwargs.get('version',    None)
        confirm = kwargs.get('confirm',    False)

        query = {'detector':detname}
        if ctype != defs['ctype']    : query['ctype']    = ctype
        if run   != defs['run']      : query['run']      = run
        if vers  != defs['version']  : query['version']  = vers
        #if tsec  != defs['time_sec'] : query['time_sec'] = tsec
        if is_in_command_line('-s', '--time_sec')   : query['time_sec'] = tsec 
        if is_in_command_line('-t', '--time_stamp') : query['time_stamp'] = tstamp 

        logger.info('query: %s' % str(query))

        docs = dbu.find_docs(col, query)
        if docs is None or docs.count() == 0 :
            logger.warning('Can not find document for query: %s' % str(query))
            return

        for i,doc in enumerate(docs) :
            msg = '  deldoc %2d:'%i + doc['time_stamp'] + ' ' + str(doc['time_sec'])\
                + ' %s'%doc['ctype'].ljust(16) + ' %4d'%doc['run'] + ' ' + str(doc['id_data'])
            logger.info(msg)
            if confirm : 
                dbu.delete_document_from_collection(col, doc['_id'])
                dbu.del_document_data(doc, fs)

        if not confirm : dbu.request_confirmation()


    def add(self) :
        """Adds calibration constants to database from file.
        """
        kwargs = self.kwargs
        fname = kwargs.get('iofname', 'None')
        ctype = kwargs.get('ctype', 'None')
        verb  = self.loglevel == 'DEBUG'
        assert os.path.exists(fname), 'File "%s" DOES NOT EXIST' % fname

        ext = os.path.splitext(fname)[-1]
        data = gu.load_textfile(fname, verb=verb) if ctype == 'geometry' else\
               np.load(fname) if ext == '.npy' else\
               load_txt(fname)

        dbu.insert_calib_data(data, **kwargs)


    def get(self) :
        """Gets constans from DB and saves them in file.
        """
        mode, kwargs = self.mode, self.kwargs
        defs   = self.defs
        host   = kwargs.get('host', None)
        port   = kwargs.get('port', None)
        exp    = kwargs.get('experiment', None)
        det    = kwargs.get('detector', None)
        ctype  = kwargs.get('ctype', None)
        run    = kwargs.get('run', None)
        run_end = kwargs.get('run_end', None)
        tsec   = kwargs.get('time_sec', None)   if is_in_command_line('-s', '--time_sec')   else None
        tstamp = kwargs.get('time_stamp', None) if is_in_command_line('-t', '--time_stamp') else None
        vers   = kwargs.get('version', None)
        prefix = kwargs.get('iofname', None)
        verb   = self.loglevel == 'DEBUG'

        db_det, db_exp, colname, query = dbu.dbnames_collection_query(det, exp, ctype, run, tsec, vers)
        logger.debug('get: %s %s %s %s' % (db_det, db_exp, colname, str(query)))
        dbname = db_det if exp is None else db_exp

        client = self.client()
        if not self.check_database(client, dbname) : return

        db, fs = dbu.db_and_fs(client, dbname)
        colnames = dbu.collection_names(db)

        if not(colname in colnames) : # dbu.collection_exists(db, colname)
            logger.warning('db "%s" does not have collection "%s"' % (db.name, str(colname)))
            return
            
        col = dbu.collection(db,colname)

        logger.debug('Search document in db "%s" collection "%s"' % (dbname,colname))

        doc = dbu.find_doc(col, query)
        if doc is None :
            logger.warning('Can not find document for query: %s' % str(query))
            return
            
        logger.debug('get doc:', doc)

        data = dbu.get_data_for_doc(fs, doc)
        if data is None :
            logger.warning('Can not load data for doc: %s' % str(doc))
            return

        if prefix is None : prefix = dbu.out_fname_prefix(**doc)

        dbu.save_doc_and_data_in_file(doc, data, prefix, control={'data' : True, 'meta' : True})


    def host_port_dbname_fname(self) :
        kwargs = self.kwargs
        host   = kwargs.get('host', None)
        port   = kwargs.get('port', None)
        dbname = kwargs.get('dbname', None)
        fname  = kwargs.get('iofname', None)
        return host, port, dbname, fname


    def exportdb(self) :
        """Exports database. Equivalent to: mongodump -d <dbname> -o <filename>
           mongodump --host psanaphi105 --port 27017 --db calib-cxi12345 --archive=db.20180122.arc 
        """
        host, port, dbname, fname = self.host_port_dbname_fname()

        tstamp = gu.str_tstamp(fmt='%Y-%m-%dT%H-%M-%S')
        fname = 'cdb-%s-%s.arc' % (tstamp, dbname) if fname is None else fname

        dbu.exportdb(host, port, dbname, fname)


    def importdb(self) :
        """Imports database. Equivalent to: mongorestore -d <dbname> --archive <filename>
           mongorestore --archive cdb-2018-03-09T10-19-30-cdb-cxix25115.arc --db calib-cxi12345
        """
        host, port, dbname, fname = self.host_port_dbname_fname()

        dbu.importdb(host, port, dbname, fname)


    def test(self) :
        host, port, dbname, fname = self.host_port_dbname_fname()
        dbnames = dbu.database_names(self.client())
        logger.info('dbnames: %s' % ', '.join(dbnames))


    def _warning(self) :
        logger.warning('MDB_CLI: TBD for mode: %s' % self.mode)


    def dispatcher(self) :
        mode = self.mode
        logger.debug('Mode: %s' % mode)
        if   mode == 'print'  : self.print_content()
        elif mode == 'convert': self.convert()
        elif mode == 'deldoc' : self.deldoc()
        elif mode == 'delcol' : self.delcol()
        elif mode == 'deldb'  : self.deldb()
        elif mode == 'delall' : self.delall()
        elif mode == 'add'    : self.add()
        elif mode == 'get'    : self.get()
        elif mode == 'export' : self.exportdb()
        elif mode == 'import' : self.importdb()
        elif mode == 'test'   : self.test()
        else : logger.warning('Non-implemented command mode "%s"\n  Known modes: %s' % (mode,', '.join(MODES)))

#------------------------------

def cdb(parser) :
    """Calibration Data Base Command Line Interface
    """
    MDB_CLI(parser)

#------------------------------

if __name__ == "__main__" :
    import sys
    sys.exit('See example in app/cdb.py')

#------------------------------

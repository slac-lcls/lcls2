#------------------------------
"""
Created on 2018-02-23 by Mikhail Dubrovin
"""
#------------------------------

import os
from psana.pyalgos.generic.Utils import print_kwargs, print_parser
import psana.pyalgos.generic.Utils as gu
from psana.pyalgos.generic.NDArrUtils import print_ndarr
#import psana.pscalib.calib.CalibConstants as cc
from psana.pscalib.calib.NDArrIO import load_txt, save_txt

import psana.pscalib.calib.MDBUtils as dbu # insert_constants, time_and_timestamp
import numpy as np

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
          -  verbose   
          -  iofname
          -  comment
          -  dbname
        """
        (popts, pargs) = parser.parse_args()
        #args = pargs
        #opts = vars(popts)
        #defs = vars(parser.get_default_values())
        #host = kwargs.get('host', None),

        self.mode = mode = pargs[0] if len(pargs)>0 else 'print'
        assert mode in ('print', 'add', 'get', 'convert', 'delete', 'export', 'import'),\
                       'Not allowed command mode "%s"' % mode 

        kwargs = vars(popts)

        time_sec, time_stamp = dbu.time_and_timestamp(**kwargs)
        kwargs['time_sec'] = str(time_sec)
        kwargs['time_stamp'] = time_stamp

        self.kwargs = kwargs

        if popts.verbose : 
            print_parser(parser)
            print_kwargs(kwargs)


    def client(self) :
        kwargs = self.kwargs
        host = kwargs.get('host', None)
        port = kwargs.get('port', None)
        print('\nMongoDB client host:%s port:%d' % (host, port))
        return dbu.connect_to_server(host, port)


    def print_content(self) :
        kwargs = self.kwargs
        exp  = kwargs.get('experiment', None)
        det  = kwargs.get('detector', None)
        client = self.client()

        if   exp is not None : print(dbu.database_info(client, exp, level=3))
        elif det is not None : print(dbu.database_info(client, det, level=3))
        else                 : print(dbu.client_info(client, level=2))


    def convert(self) :
        """Converts LCLS1 experiment calib directory to calibration database.
        """
        import psana.pscalib.calib.MDBConvertLCLS1 as cu
        kwargs = self.kwargs
        exp = kwargs.get('experiment', None)
        cu.scan_calib_for_experiment(exp, **kwargs)


    def delete(self) :
        """Deletes database.
        """
        mode, kwargs = self.mode, self.kwargs
        exp    = kwargs.get('experiment', None)
        det    = kwargs.get('detector', None)
        dbname = kwargs.get('dbname', None)

        if dbname is None :
            name = exp if not (exp is None) else det
            if name is None :
                print('To delete database experiment or detector has to be specified')
                return
            dbname = dbu.db_prefixed_name(name)

        client = self.client()

        if mode == 'delete' :
            print('Databases before delete:\n%s' % str(dbu.database_names(client)))
            dbu.delete_database(client, dbname)
            print('Databases after "delete" %s:\n%s' % (dbname, str(dbu.database_names(client))))


    def add(self) :
        """Adds calibration constants to database from file.
        """
        kwargs = self.kwargs
        fname = kwargs.get('iofname', 'None')
        ctype = kwargs.get('ctype', 'None')
        assert os.path.exists(fname), 'File "%s" DOES NOT EXIST !' % fname

        ext = os.path.splitext(fname)[-1]

        data = gu.load_textfile(fname, verb=False) if ctype == 'geometry' else\
               np.load(fname) if ext == '.npy' else\
               load_txt(fname)

        dbu.insert_calib_data(data, **kwargs)
        #dbu.insert_constants(data, d['experiment'], d['detector'], d['ctype'], d['run'],\
        #                     d['time_sec_or_stamp'], d['version'], **kwargs) :


    def get(self) :
        """Saves in file calibration constants from database.
        """
        kwargs = self.kwargs
        host   = kwargs.get('host', None)
        port   = kwargs.get('port', None)
        exp    = kwargs.get('experiment', None)
        det    = kwargs.get('detector', None)
        ctype  = kwargs.get('ctype', None)
        verb   = kwargs.get('verbose', False)
        fname  = kwargs.get('iofname', None)

        client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
            dbu.connect(host=host, port=port, experiment=exp, detector=det, verbose=verb)
            #dbu.connect(host, port, exp, det, verb) 

        query={'detector':det, 'ctype':ctype}

        fs, doc = fs_exp, dbu.find_doc(col_exp, query)
        if doc is None :
            fs, doc = fs_det, dbu.find_doc(col_det, query)
            if doc is None :
                print('Can not find document for query: %s' % str(query))
                return

        data = dbu.get_data_for_doc(fs, doc)
        if data is None :
                print('Can not load data for doc: %s' % str(doc))
                return

        if fname is None : fname='clb-%s-%s-%s.npy' % (expname, det, ctype)

        if ctype == 'geometry' : 
            gu.save_textfile(data, fname, mode='w', verb=verb)
        elif os.path.splitext(fname)[1] == '.npy' : 
            np.save(fname, data, allow_pickle=False)
        else : 
            #np.savetxt(fname, data, fmt='%.2f')
            save_txt(fname, data, fmt='%.2f')

        if verb : 
            print('Save constants from db in file: %s' % fname)
            print_ndarr(data, 'nda:')


    def exec_command(self, cmd) :
        from psana.pscalib.proc.SubprocUtils import subproc
        print('Execute shell command: %s' % cmd)
        if not gu.shell_command_is_available(cmd.split()[0], verb=True) : return
        out,err = subproc(cmd, env=None, shell=False, do_wait=True)
        print('err: %s\nout: %s' % (err,out))


    def exportdb(self) :
        """Exports database. Equivalent to: mongodump -d <dbname> -o <filename>
           mongodump --host psanaphi105 --port 27017 --db calib-cxi12345 --archive=db.20180122.arc 
        """
        kwargs = self.kwargs
        host   = kwargs.get('host', None)
        port   = kwargs.get('port', None)
        dbname = kwargs.get('dbname', None)
        ofname = kwargs.get('iofname', None)

        dbnames = dbu.database_names(self.client())
        if not (dbname in dbnames) :
            print('WARNING: --dbname %s is not available in the list:\n%s' % (dbname, dbnames))
            return

        tstamp = gu.str_tstamp(fmt='%Y-%m-%dT%H-%M-%S')
        fname = 'cdb-%s-%s.arc' % (tstamp, dbname) if ofname is None else ofname

        cmd = 'mongodump --host %s --port %s --db %s --archive %s' % (host, port, dbname, fname)
        self.exec_command(cmd)


    def importdb(self) :
        """Imports database. Equivalent to: mongorestore -d <dbname> --archive <filename>
           mongorestore --archive cdb-2018-03-09T10-19-30-cdb-cxix25115.arc --db calib-cxi12345
        """
        kwargs = self.kwargs
        host   = kwargs.get('host', None)
        port   = kwargs.get('port', None)
        dbname = kwargs.get('dbname', None)
        fname  = kwargs.get('iofname', None)

        if fname is None :
            print('WARNING input archive file name should be specified as --iofname <fname>')
            return 

        dbnames = dbu.database_names(self.client())
        if dbname in dbnames :
            print('WARNING: --dbname %s is already available in the list:\n%s' % (dbname, dbnames))
            return

        cmd = 'mongorestore --host %s --port %s --db %s --archive %s' % (host, port, dbname, fname)
        self.exec_command(cmd)


    def _warning(self) :
        print('MDB_CLI: TBD for mode: %s' % self.mode)


    def dispatcher(self) :
        mode = self.mode
        #print('Mode: %s' % mode)
        if   'print'   in mode : self.print_content()
        elif 'convert' in mode : self.convert()
        elif 'delete'  in mode : self.delete()
        elif 'add'     in mode : self.add()
        elif 'get'     in mode : self.get()
        elif 'export'  in mode : self.exportdb()
        elif 'import'  in mode : self.importdb()
        #elif 'get'     in mode : self._warning()
        else : print('Not allowed command mode "%s"' % mode)

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

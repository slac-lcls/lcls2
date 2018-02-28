#------------------------------
"""
Created on 2018-02-23 by Mikhail Dubrovin
"""
#------------------------------

import os
import sys
from psana.pyalgos.generic.Utils import print_kwargs, print_parser
#import psana.pscalib.calib.CalibConstants as cc

from psana.pscalib.calib.MDBUtils import insert_constants

#------------------------------

class MDB_CLI :

    def __init__(self, parser) : 
        self.unpack(parser)
        self.dispatcher()

    def unpack(self, parser) :
        """
          -  host      
          -  port      
          -  experiment
          -  detector  
          -  ctype     
          -  run       
          -  time_stamp
          -  version   
          -  verbose   
          -  iofname
        """  

        (popts, pargs) = parser.parse_args()
        #args = pargs
        #opts = vars(popts)
        #defs = vars(parser.get_default_values())
        #host = kwargs.get('host', None),

        self.mode = mode = pargs[0] if len(pargs)>0 else 'print'
        assert mode in ('print', 'save', 'get'), 'Not allowed command mode "%s"' % mode 

        self.kwargs = vars(popts)
        if popts.verbose : 
            print_parser(parser)
            print_kwargs(self.kwargs)

    def _warning(self) :
        print('MDB_CLI: TBD for mode: %s' % self.mode)


    def dispatcher(self) :
        print('Envoke dispatcher for mode: %s' % self.mode)

        mode, kwargs = self.mode, self.kwargs

        if mode == 'print' :
            self._warning()

        elif mode == 'save' :
            fname = kwargs.get('iofname', 'None')
            assert os.path.exists(fname), 'File "%s" DOES NOT EXIST !' % fname
            data = 'Just a test record'
            insert_constants(**kwargs)
            #insert_constants(data, experiment, detector, ctype, run, time_sec_or_stamp, version, **kwargs) :

        elif mode == 'get' :
            self._warning()

        else :
            print('Not allowed command mode "%s"' % mode)

    #cdb(**kwargs)

#------------------------------

def cdb(parser) :
    """Calibration Data Base Command Line Interface
    """
    MDB_CLI(parser)

#------------------------------
#------------------------------

if __name__ == "__main__" :
    sys.exit('See example in app/cdb')

#------------------------------

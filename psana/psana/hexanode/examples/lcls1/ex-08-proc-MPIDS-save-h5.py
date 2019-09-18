#!@PYTHON@

####!/usr/bin/env python

""" Example of hexanode LCLS data processing using MPI and creating "small data" hdf5 file.
"""
#------------------------------
import sys
from expmon.HexDataPreProc import preproc_data
#------------------------------

def usage():
    return 'Use command: mpirun -n 8 python hexanode/examples/ex-08-proc-MPIDS-save-h5.py\n'\
           'or:\n'\
           '   bsub -o log-mpi-n02-%J.log -q psnehq -n 2 mpirun python hexanode/examples/ex-08-proc-MPIDS-save-h5.py'

#------------------------------

if __name__ == "__main__" :

    print 50*'_'
    print usage()

    # Parameters for initialization of the data source, channels, number of events etc.
    kwargs = {'srcchs'   : {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)},
              'numchs'   : 7,
              'numhits'  : 16,
              'dsname'   : 'exp=xpptut15:run=390:smd',
              'evskip'   : 0,
              'events'   : 1200,
              'ofprefix' : './',
             }

    # Parameters of the CFD descriminator for hit time finding algotithm
    cfdpars= {'cfd_base'       :  0.,
              'cfd_thr'        : -0.05,
              'cfd_cfr'        :  0.9,
              'cfd_deadtime'   :  5.0,
              'cfd_leadingedge':  True,
              'cfd_ioffsetbeg' :  0,
              'cfd_ioffsetend' :  1000,
             }

    kwargs.update(cfdpars)
    preproc_data(**kwargs)

    sys.exit('End of %s' % sys.argv[0])

#------------------------------

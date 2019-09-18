#!@PYTHON@

####!/usr/bin/env python

""" Example of hexanode calibration using "small data" hdf5 file (fast) or xtc data (slow).
"""
#------------------------------

import sys
from expmon.HexCalib import calib_on_data

#------------------------------

def usage():
    return 'Use command: python hexanode/examples/ex-09-sort-graph-data.py'

#------------------------------

if __name__ == "__main__" :

    print 50*'_'

              #'dsname'   : 'exp=xpptut15:run=390:smd',
              #'dsname'   : './xpptut15-r0390-e001200-single-node.h5',
              #'dsname'   : './xpptut15-r0390-e001200-n02-mpi.h5',
              #'dsname'   : './xpptut15-r0390-e300000-n04-mpi.h5',
              #'dsname'   : './xpptut15-r0390-e001200-single-node.h5',

    # Parameters for initialization of the data source, channels, number of events etc.
    kwargs = {'srcchs'   : {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)},
              'numchs'   : 7,
              'numhits'  : 16,
              'dsname'   : 'exp=xpptut15:run=390:smd',
              'evskip'   : 0,
              'events'   : 1200,
              'ofprefix' : './figs-hexanode/plot',
              'calibtab' : 'calibration_table_data.txt',
              'calibcfg' : 'configuration_hex.txt',
              'plot_his' : True,
              'verbose'  : False,
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

    # On/Off graphics parameters
    plotpars={'PLOT_NHITS'         : True,
              'PLOT_TIME_CH'       : True,
              'PLOT_UVW'           : True,
              'PLOT_TIME_SUMS'     : True,
              'PLOT_CORRELATIONS'  : True,
              'PLOT_XY_COMPONENTS' : True,
              'PLOT_XY_2D'         : True,
              'PLOT_XY_RESOLUTION' : True,
              'PLOT_MISC'          : True,
              'PLOT_REFLECTIONS'   : True,
              'PLOT_PHYSICS'       : True,
             }

    kwargs.update(cfdpars) # may need it if do waveform reconstruction from xtc data

    kwargs.update(plotpars)

    calib_on_data(**kwargs)

    sys.exit('End of %s' % sys.argv[0])

#------------------------------

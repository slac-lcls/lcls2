#!/usr/bin/env python
####!@PYTHON@

""" Example of hexanode calibration using "small data" hdf5 file (fast) or xtc data (slow).
"""
#------------------------------

import sys

#------------------------------

def usage():
    return 'Use command: python hexanode/examples/ex-quad-09-sort-graph-data.py',\
           '\n  needs in /reg/d/psdm/amo/amox27716/calib/Acqiris::CalibV1/AmoEndstation.0:Acqiris.1/hex_config/0-end.data'

#------------------------------

if __name__ == "__main__" :

    print(50*'_')

              #'dsname'   : 'exp=xpptut15:run=390:smd',
              #'dsname'   : './xpptut15-r0390-e001200-single-node.h5',
              #'dsname'   : './xpptut15-r0390-e001200-n02-mpi.h5',
              #'dsname'   : './xpptut15-r0390-e300000-n04-mpi.h5',
              #'dsname'   : './xpptut15-r0390-e001200-single-node.h5',
              #
              #'dsname'   : 'exp=amox27716:run=100',
              #'dsname'   : 'amox27716-r0100-e060000-single-node.h5',
              #'dsname'   : 'amox27716-r0100-e060000-single-node.h5',

    # Parameters for initialization of the data source, channels, number of events etc.
    kwargs = {'srcchs'   : {'AmoEndstation.0:Acqiris.1':(2,3,4,5,6)},
              'numchs'   : 5,
              'numhits'  : 16,
              'dsname'   : 'amox27716-r0100-e060000-single-node.h5',
              'evskip'   : 0,
              'events'   : 60000,
              'ofprefix' : './figs-quadanode/plot',
              'calibtab' : 'calibration_table_data.txt',
              'calibcfg' : 'configuration_quad.txt',
              'plot_his' : True,
              'verbose'  : False,
             }
              #'ofprefix' : './figs-quadanode/plot-sorted',

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
              'PLOT_REFLECTIONS'   : True,
              'PLOT_PHYSICS'       : True,
              'PLOT_MISC'          : False,
             }

    kwargs.update(cfdpars)
    kwargs.update(plotpars)

    #from expmon.QuadCalib import calib_on_data
    #calib_on_data(**kwargs)

    sys.exit('End of %s' % sys.argv[0])

#------------------------------

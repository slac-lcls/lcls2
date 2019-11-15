#!/usr/bin/env python
####!@PYTHON@

""" Example of hexanode calibration using "small data" hdf5 file (fast) or xtc data (slow).
"""
#------------------------------

import sys

#------------------------------

def usage():
    return 'Use command: python psana/psana/hexanode/examples/ex-14-quad-sort-graph-data.py',\
           '\n  needs in /reg/d/psdm/amo/amox27716/calib/Acqiris::CalibV1/AmoEndstation.0:Acqiris.1/hex_config/0-end.data'

#------------------------------

if __name__ == "__main__" :

    print(50*'_')

    # Parameters for initialization of the data source, channels, number of events etc.
    #'dsname' : '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2',
    kwargs = {
              'ifname' : '/reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5',
              'detname'  : 'tmo_hexanode',
              'numchs'   : 5,
              'numhits'  : 16,
              'evskip'   : 7,
              'events'   : 60000,
              'ofprefix' : './tst',
              'run'      : 100,
              'exp'      : 'amox27716',
              'calibcfg' : '/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/configuration_quad.txt',
              'calibtab' : '/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/calibration_table_data.txt',
              'plot_his' : True,
              'save_his' : False,
              'verbose'  : False,
             }

    # Parameters of the CFD descriminator for hit time finding algotithm
    cfdpars= {'cfd_base'       :  0.,
              'cfd_thr'        : -0.05,
              'cfd_cfr'        :  0.85,
              'cfd_deadtime'   :  10.0,
              'cfd_leadingedge':  True,
              'cfd_ioffsetbeg' :  1000,
              'cfd_ioffsetend' :  2000,
              'cfd_wfbinbeg'   :  6000,
              'cfd_wfbinend'   : 22000,
             }

    # On/Off graphics parameters
    plotpars={'PLOT_NHITS'         : True,
              'PLOT_TIME_CH'       : False,
              'PLOT_REFLECTIONS'   : False,
              'PLOT_UVW'           : True,
              'PLOT_TIME_SUMS'     : True,
              'PLOT_CORRELATIONS'  : True,
              'PLOT_XY_COMPONENTS' : False,
              'PLOT_PHYSICS'       : False,
              'PLOT_MISC'          : False,
              'PLOT_XY_2D'         : True,
             }

    kwargs.update(cfdpars)
    kwargs.update(plotpars)

    from psana.pyalgos.generic.Utils import str_kwargs
    print(str_kwargs(kwargs, title='app input parameters:'))

    from psana.hexanode.QuadCalib import calib_on_data
    calib_on_data(**kwargs)

    sys.exit('End of %s' % sys.argv[0])

#------------------------------

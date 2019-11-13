#------------------------------
"""
Created on 2019-11-13 by Mikhail Dubrovin
"""
#------------------------------

#import logging
#logger = logging.getLogger(__name__)

import sys
from psana.graphqt.H5VMain import hdf5explorer

#------------------------------

def usage():
    return 'command examples for app %s\n'%sys.argv[0]\
         + '  hdf5explorer\n'\
         + '  hdf5explorer <hdf5-file-name>\n'\
         + '  hdf5explorer /reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5\n'\
         + '  hdf5explorer /reg/g/psdm/detector/calib/jungfrau/jungfrau-171113-154920171025-3d00fb.h5'
 
#------------------------------

def hdf5explorer_gui() :
    """hdf5explorer
    """
    parser = input_option_parser()

    if len(sys.argv) == 1 :
        #print(80*'_')
        #parser.print_help()
        print(80*'_')
        parser.print_usage()
        print(80*'_')

    (popts, pargs) = parser.parse_args() # TRICK! this line allows -h or --help potion !!!
    hdf5explorer(parser)

    #opts = vars(popts)
    #kwargs = opts
    #print_kwargs(kwargs)
    #print_parser(parser)
    #hdf5explorer(**kwargs)

#------------------------------

def input_option_parser() :

    from optparse import OptionParser

    d_experiment = 'exp12345'
    d_detector   = 'detector_1234'
    d_loglevel   = 'INFO'
    d_logdir     = './cm-logger'

    h_experiment = 'experiment name, default = %s' % d_experiment
    h_detector   = 'detector name, default = %s' % d_detector
    h_loglevel   = 'logging level, default = %s' % d_loglevel
    h_logdir     = 'logger directory, default = %s' % d_logdir

    parser = OptionParser(description='Calibration manager UI', usage=usage())

    parser.add_option('-d', '--detector',   default=d_detector,   action='store', type='string', help=h_detector)
    parser.add_option('-e', '--experiment', default=d_experiment, action='store', type='string', help=h_experiment)
    parser.add_option('-l', '--loglevel',   default=d_loglevel,   action='store', type='string', help=h_loglevel)
    parser.add_option('-L', '--logdir',     default=d_logdir,     action='store', type='string', help=h_logdir)
    #parser.add_option('-v', '--verbose',    default=d_verbose,    action='store_false',          help=h_verbose)

    return parser
  
#------------------------------

if __name__ == "__main__" :
    hdf5explorer_gui()
    sys.exit(0)

#------------------------------

#------------------------------
"""
Created on 2018-02-26 by Mikhail Dubrovin
"""
#------------------------------

import sys
import psana.pscalib.calib.CalibConstants as cc
from psana.graphqt.CMWMain import calibman, logging

LEVEL_NAMES = ', '.join(list(logging._levelToName.values()))

#------------------------------

def usage():
    return 'command examples for app %s\n'%sys.argv[0]\
         + '  calibman\n'\
         + '  calibman -u <username> -p <password>\n'\
         + '  calibman --host=psanagpu115 --port=27017\n'\
         + '  calibman --host=psanaphi103 -l DEBUG -L cm-log'
    #return '%s - TBD' % (sys._getframe().f_code.co_name)
 
#------------------------------

def calibman_gui() :
    """Calibration Data Base GUI
    """
    parser = input_option_parser()

    if len(sys.argv) == 1 :
        print(80*'_')
        parser.print_help()
        print(80*'_')
        parser.print_usage()
        print(80*'_')
        #msg = 'WARNING: COMMAND WITH ALL DEFAULT PARAMETERS IS USELESS...'
        #print(msg)
        #sys.exit(msg)

    (popts, pargs) = parser.parse_args() # TRICK! this line allows -h or --help potion !!!
    calibman(parser)

    #opts = vars(popts)
    #kwargs = opts
    #print_kwargs(kwargs)
    #print_parser(parser)
    #calibman(**kwargs)

#------------------------------

def input_option_parser() :

    from optparse import OptionParser

    d_host       = cc.HOST
    d_port       = cc.PORT
    d_user       = cc.USERNAME
    d_upwd       = ''
    d_experiment = 'exp12345'
    d_detector   = 'detector_1234'
    d_loglevel   = 'INFO'
    d_logdir     = './cm-logger'

    h_host       = 'DB host, default = %s' % d_host
    h_port       = 'DB port, default = %s' % d_port
    h_user   = 'username to access DB, default = %s' % d_user
    h_upwd   = 'password, default = %s' % d_upwd
    h_experiment = 'experiment name, default = %s' % d_experiment
    h_detector   = 'detector name, default = %s' % d_detector
    h_loglevel   = 'logging level from list (%s), default = %s' % (LEVEL_NAMES, d_loglevel)
    h_logdir     = 'logger directory, default = %s' % d_logdir

    parser = OptionParser(description='Calibration manager UI', usage=usage())

    parser.add_option('--host',             default=d_host,       action='store', type='string', help=h_host)
    parser.add_option('--port',             default=d_port,       action='store', type='string', help=h_port)
    parser.add_option('-u', '--user',       default=d_user,       action='store', type='string', help=h_user)
    parser.add_option('-p', '--upwd',       default=d_upwd,       action='store', type='string', help=h_upwd)
    parser.add_option('-d', '--detector',   default=d_detector,   action='store', type='string', help=h_detector)
    parser.add_option('-e', '--experiment', default=d_experiment, action='store', type='string', help=h_experiment)
    parser.add_option('-l', '--loglevel',   default=d_loglevel,   action='store', type='string', help=h_loglevel)
    parser.add_option('-L', '--logdir',     default=d_logdir,     action='store', type='string', help=h_logdir)
    #parser.add_option('-v', '--verbose',    default=d_verbose,    action='store_false',          help=h_verbose)

    return parser
  
#------------------------------

if __name__ == "__main__" :
    calibman_gui()
    sys.exit(0)

#------------------------------

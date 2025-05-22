#------------------------------
"""
Created on 2019-01-25 by Mikhail Dubrovin
"""
import os
#------------------------------

import sys
from psdaq.control_gui.CGWMain import proc_control_gui, logging
from psdaq.control_gui.CGConfigDBUtils import URI_CONFIGDB

LEVEL_NAMES = ', '.join(list(logging._levelToName.values()))

#------------------------------

def usage():
    return 'command examples for app %s\n'%sys.argv[0]\
         + '  control_gui\n'\
         + '  control_gui -l INFO -L log-daq-control'\
         + '  control_gui --loglevel=INFO --logdir=log-daq-control'
    #return '%s - TBD' % (sys._getframe().f_code.co_name)
 
#------------------------------

def control_gui() :
    """Launch DAQ Control GUI
    """
    parser = input_option_parser()

    if len(sys.argv) == 1 :
        print(80*'_')
        parser.print_help()
        print(80*'_')
        parser.print_usage()
        print(80*'_')

    (popts, pargs) = parser.parse_args() # TRICK! this line allows -h or --help option !!!
    proc_control_gui(parser)

    #opts = vars(popts)
    #kwargs = opts
    #print_kwargs(kwargs)
    #print_parser(parser)
    #proc_control_gui(**kwargs)

#------------------------------

def input_option_parser() :

    from optparse import OptionParser

    d_platform   = 6 # [0-7]
    d_host       = 'localhost'
    d_timeout    = 60000 # ms
    d_loglevel   = 'DEBUG'
    d_logdir     = None # '.' or './cm-logger' etc.
    d_expname    = 'tmo12345'
    d_uris       = URI_CONFIGDB # 'https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/'
    d_expert     = False  
    d_user       = 'tmoopr'
    d_password   = os.getenv('CONFIGDB_AUTH')

    h_platform   = 'platform in range [0,7], default = %s' % d_platform
    h_host       = 'control host, default = %s' % d_host
    h_timeout    = 'timeout [ms], default = %s' % d_timeout
    h_loglevel   = 'logging level from list (%s), default = %s' % (LEVEL_NAMES, d_loglevel)
    h_logdir     = 'logger directory, default = %s' % d_logdir
    h_expname    = 'experiment name, default = %s' % d_expname
    h_uris       = 'configuration DB URI suffix, default = %s' % d_uris
    h_expert     = 'force start gui in expert mode, default = %s' % d_expert
    h_user       = 'user login name, default = %s' % d_user
    h_password   = 'password for interaction with configuration DB, default = *****'

    parser = OptionParser(description='DAQ Control GUI', usage=usage())

    parser.add_option('-p', '--platform', default=d_platform, action='store', type='int',    help=h_platform)
    parser.add_option('-H', '--host',     default=d_host,     action='store', type='string', help=h_host)
    parser.add_option('-t', '--timeout',  default=d_timeout,  action='store', type='int',    help=h_timeout)
    parser.add_option('-l', '--loglevel', default=d_loglevel, action='store', type='string', help=h_loglevel)
    parser.add_option('-L', '--logdir',   default=d_logdir,   action='store', type='string', help=h_logdir)
    parser.add_option('-e', '--expname',  default=d_expname,  action='store', type='string', help=h_expname)
    parser.add_option('-u', '--uris',     default=d_uris,     action='store', type='string', help=h_uris)
    parser.add_option('-E', '--expert',   default=d_expert,   action='store_true',           help=h_expert)
    parser.add_option(      '--user',     default=d_user,     action='store', type='string', help=h_user)
    parser.add_option(      '--password', default=d_password, action='store', type='string', help=h_password)

    return parser
  
#------------------------------

if __name__ == "__main__" :
    control_gui()
    sys.exit(0)

#------------------------------

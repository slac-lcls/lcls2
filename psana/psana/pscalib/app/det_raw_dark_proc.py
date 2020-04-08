#!/usr/bin/env python

#----------
import logging
logger = logging.getLogger(__name__)
import sys

SCRNAME = sys.argv[0].rsplit('/')[-1]

DICT_NAME_TO_LEVEL = logging._nameToLevel
STR_LEVEL_NAMES = ', '.join(DICT_NAME_TO_LEVEL.keys())

#----------

def init_logger(loglev_name) :
    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s' # '%(message)s'
    fmt='[%(levelname).1s] L%(lineno)04d : %(message)s'
    logging.basicConfig(format=fmt, datefmt='%Y-%m-%dT%H:%M:%S', level=logging._nameToLevel[loglev_name])
    logger.debug('Logger is initialized for level %s' % loglev_name)

#----------

def usage(mode=0) :
    if mode == 1 : return 'Proceses detector raw n-d array and saves results in files.'
    else : return\
           '       %s -d <dataset> [-s <source>] [-f <file-name-template>]' % SCRNAME +\
           ' [-n <events-collect>] [-m <events-skip>] [-v 7] [-p 1] [-v 7] ...'+\
           '\n  -v, -S control bit-words stand for 1/2/4/8/16/32/... - ave/rms/status/mask/max/min/sta_int_lo/sta_int_hi'+\
           '\nEx.1:  %s -e amox23616 -r 104 -s xtcav -o nda.expand -n 20 -m 0 -v 7 -p 1' % SCRNAME +\
           '\nEx.2:  %s -e mecj5515 -r 102 -s MecTargetChamber.0:Cspad2x2.1,MecTargetChamber.0:Cspad2x2.2 -o nda-#exp-#run-#src-#type.txt -n 10' % SCRNAME +\
           '\nEx.3:  %s -e amo42112 -r 120 -s AmoBPS.0:Opal1000.0 -n 5 -o nda-#exp-#run-peds-#type-#src.txt' % SCRNAME 

#----------

def command_line_parser() :

    import argparse

    d_expnam = 'amox23616'
    d_runnum = 104
    d_source = 'xtcav' # 'CxiDs2.0:Cspad.0,xtcav' or list of names
    d_ifname = '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0104-e000400-xtcav-v2.xtc2'
    d_ofname = 'nda-#exp-#run-#src-#evts-#type-#date-#time-#fid-#sec-#nsec.txt'
    d_events = 1000
    d_evskip = 0
    d_intlow = 0
    d_inthig = 16000
    d_rmslow = 0
    d_rmshig = 16000
    d_fraclm = 0.1
    d_nsigma = 6.0
    d_plotim = 0      
    d_verbos = 7
    d_savebw = 0o377
    d_intnlo = 6.0
    d_intnhi = 6.0
    d_rmsnlo = 6.0
    d_rmsnhi = 6.0
    d_evcode = None
    d_loglev = 'DEBUG'
   
    h_expnam='dataset name, default = %s' % d_expnam
    h_runnum='run number, default = %s' % d_runnum
    h_source='input ndarray source name, default = %s' % d_source
    h_ifname='input file name, default = %s' % d_ifname
    h_ofname='output file name template, default = %s' % d_ofname
    h_events='number of events to collect, default = %s' % d_events
    h_evskip='number of events to skip, default = %s' % d_evskip
    h_intlow='intensity low limit, default = %s' % d_intlow
    h_inthig='intensity high limit, default = %s' % d_inthig
    h_rmslow='rms low limit, default = %s' % d_rmslow
    h_rmshig='rms high limit, default = %s' % d_rmshig
    h_fraclm='allowed fraction limit, default = %s' % d_fraclm
    h_nsigma='number of sigma for gated average, default = %s' % d_nsigma
    h_plotim='control bit-word to plot images, default = %s' % d_plotim
    h_verbos='control bit-word for verbosity, default = %s' % d_verbos
    h_savebw='control bit-word to save arrays, default = %s' % d_savebw
    h_intnlo='number of sigma from mean for low  limit on INTENSITY, default = %s' % d_intnlo
    h_intnhi='number of sigma from mean for high limit on INTENSITY, default = %s' % d_intnhi
    h_rmsnlo='number of sigma from mean for low  limit on RMS, default = %s' % d_rmsnlo
    h_rmsnhi='number of sigma from mean for high limit on RMS, default = %s' % d_rmsnhi
    h_evcode='comma separated event codes for selection as OR combination, any negative code inverts selection, default = %s' % str(d_evcode)
    h_loglev='logging level name, one of %s, default = %s' % (STR_LEVEL_NAMES, str(d_loglev))

    parser = argparse.ArgumentParser(description=usage(1), usage=usage())

    parser.add_argument('-e', '--expnam', default=d_expnam, type=str,   help=h_expnam)
    parser.add_argument('-r', '--runnum', default=d_runnum, type=int,   help=h_runnum)
    parser.add_argument('-s', '--source', default=d_source, type=str,   help=h_source)
    parser.add_argument('-f', '--ifname', default=d_ifname, type=str,   help=h_ifname)
    parser.add_argument('-o', '--ofname', default=d_ofname, type=str,   help=h_ofname)
    parser.add_argument('-n', '--events', default=d_events, type=int,   help=h_events)
    parser.add_argument('-m', '--evskip', default=d_evskip, type=int,   help=h_evskip)
    parser.add_argument('-b', '--intlow', default=d_intlow, type=float, help=h_intlow)
    parser.add_argument('-t', '--inthig', default=d_inthig, type=float, help=h_inthig)
    parser.add_argument('-B', '--rmslow', default=d_rmslow, type=float, help=h_rmslow)
    parser.add_argument('-T', '--rmshig', default=d_rmshig, type=float, help=h_rmshig)
    parser.add_argument('-F', '--fraclm', default=d_fraclm, type=float, help=h_fraclm)
    parser.add_argument('-g', '--nsigma', default=d_nsigma, type=float, help=h_nsigma)
    parser.add_argument('-p', '--plotim', default=d_plotim, type=int,   help=h_plotim)
    parser.add_argument('-v', '--verbos', default=d_verbos, type=int,   help=h_verbos)
    parser.add_argument('-S', '--savebw', default=d_savebw, type=int,   help=h_savebw)
    parser.add_argument('-D', '--intnlo', default=d_intnlo, type=float, help=h_intnlo)
    parser.add_argument('-U', '--intnhi', default=d_intnhi, type=float, help=h_intnhi)
    parser.add_argument('-L', '--rmsnlo', default=d_rmsnlo, type=float, help=h_rmsnlo)
    parser.add_argument('-H', '--rmsnhi', default=d_rmsnhi, type=float, help=h_rmsnhi)
    parser.add_argument('-c', '--evcode', default=d_evcode, type=str,   help=h_evcode)
    parser.add_argument('-l', '--loglev', default=d_loglev, type=str,   help=h_loglev)
 
    return parser

#----------

if __name__ == "__main__" :

    print('Command   :', ' '.join(sys.argv))
    print('Arguments :')
    parser = command_line_parser()
    args = parser.parse_args()
    for k,v in args.__dict__.items() : print('   %s : %s' % (k, str(v)))

    init_logger(args.loglev)

    #sys.exit('TEST EXIT %s' % SCRNAME)
    ##================================

    from psana.pscalib.calibprod.DetRawDarkProc import detectors_dark_proc
    detectors_dark_proc(parser)

    #print(usage())

    sys.exit('END OF %s' % SCRNAME)

#----------

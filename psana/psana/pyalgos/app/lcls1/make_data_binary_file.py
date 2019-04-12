####!/usr/bin/env python
####!@PYTHON@

#------------------------------
import os
import sys
from optparse import OptionParser

import psana
import numpy as np

from pyimgalgos.GlobalUtils import print_ndarr
from pyimgalgos.Entropy import entropy

#------------------------------

def make_data_binary_file(dsname   = 'exp=xpptut15:run=54',\
                          dname    = 'CxiDs2.0:Cspad.0',\
                          nevs     = 1000000,\
                          nskip    = 0,\
                          do_calib = False,\
                          ofname   = 'data.bin',\
                          verbos   = True) :

    ds  = psana.DataSource(dsname)
    det = psana.Detector(dname)
    #run = ds.runs().next()

    nev_min = nskip
    nev_max = nskip + nevs

    f = open(ofname,'wb')

    for nevt, evt in enumerate(ds.events()):
       if nevt%100 == 0 : print 'Event %d'%nevt
       if nevt < nev_min : continue
       if nevt >= nev_max : break
       data = det.calib(evt) if do_calib else\
              det.raw(evt)
       if data is None: continue
       if do_calib: data = data.astype(np.int16)

       if verbos :
           ent = entropy(data)
           msg = '%4d: data entropy=%.3f' % (nevt, ent) 
           print_ndarr(data, name=msg, first=0, last=10)

       #data.tofile(f)
       f.write(data.tobytes())

    f.close()
    print '%d datasets saved file %s' % (nevt, ofname)

#------------------------------

def read_data_from_binary_file(ifname   = 'data.bin',\
                               npixels  = 32*185*388,\
                               dtype    = np.int16,\
                               verbos   = True) :

    """Test read/unpack binary file.
       Binary file does not have shape, so image size in pixels and data type should be provided.
    """

    print 'Read file %s' % ifname

    BUF_SIZE_BYTE = npixels*2

    f = open(ifname,'rb')
    buf = f.read()
    f.close()
    nmax = len(buf)/BUF_SIZE_BYTE
    print 'len(buf)', len(buf), 'nmax', nmax

    for nevt in range(nmax) :
        nda = np.frombuffer(buf, dtype=dtype, count=npixels, offset=nevt*BUF_SIZE_BYTE)
        print_ndarr(nda, name='%4d nda'%(nevt), first=0, last=10)

#------------------------------
#------------------------------

    def print_raw(args, opts, defs) :
        print 'Command:', ' '.join(sys.argv)
        print '  opts: ', opts
        print '  defs: ', defs
        print '  args: ', args

#------------------------------

def print_pars(args, opts, defs) :
    """Prints input parameters"""
    print 'Command:', ' '.join(sys.argv) +\
          '\nwith argument list %s and optional parameters:\n' % str(args) +\
          '<key>      <value>              <default>'
    for k,v in opts.items() :
        print '%s %s %s' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))

#------------------------------

def do_work(parser) :

    (popts, pargs) = parser.parse_args()
    args = pargs
    opts = vars(popts)
    defs = vars(parser.get_default_values())

    #print_raw(args, opts, defs)

    if popts.vrb : print_pars(args, opts, defs)
    dsname = popts.dsn if popts.dsn is not None else 'exp=%s:run=%s' % (popts.exp, popts.run)
    #print 'dsname: %s' % dsname

    if popts.bsz == 0 : 
        make_data_binary_file(dsname,\
                              dname    = popts.src,\
                              nevs     = popts.nev,\
                              nskip    = popts.nsk,\
                              do_calib = popts.clb,\
                              ofname   = popts.fnm,\
                              verbos   = popts.vrb)
    else :
        read_data_from_binary_file(ifname  = popts.fnm,\
                                   npixels = popts.bsz,\
                                   dtype   = np.int16,\
                                   verbos  = popts.vrb)

#------------------------------

def usage() :
    return '\n\nCommand to run app:\n' +\
           '\n  %prog' +\
           ' -d <data-set-name> -e <experiment> -r <run-number> -s <source-name> -n <n-events-to-read>' +\
           ' -m <m-events-to-skip> -f <binary-file-name> -b <bytes-per-event-to-read>' +\
           '\n  use option -d or -e and -r to define dataset, and option -b to test read data' +\
           '\n\n  Examples:' +\
           '\n  %prog -d exp=cxitut13:run=10 -s CxiDs1.0:Cspad.0 -n 20 -f data.bin' +\
           '\n  %prog -e cxif5315 -r 129 -s CxiDs2.0:Cspad.0 -n 10 -f data.bin' +\
           '\n  %prog -d exp=cxif5315:run=129 -s CxiDs2.0:Cspad.0 -n 10 -f data-cxif5315-r129-cspad-dark.bin' +\
           '\n  %prog -d exp=cxif5315:run=169 -s CxiDs2.0:Cspad.0 -n 10 -f data-cxif5315-r169-cspad-raw.bin' +\
           '\n  %prog -d exp=cxif5315:run=169 -s CxiDs2.0:Cspad.0 -n 10 -f data-cxif5315-r169-cspad-calib-fde.bin -c' +\
           '\n  %prog -d exp=amo86615:run=197 -s Camp.0:pnCCD.0   -n 10 -f data-amo86615-r197-pnccd-calib-spi.bin -c' +\
           '\n  %prog -d exp=cxitut13:run=10  -s CxiDs1.0:Cspad.0 -n 10 -f data-cxitut13-r10-cspad-calib-cryst.bin -c' +\
           '\n\nCommand to check saved file (if option -b is set):\n' +\
           '\n  %prog -b <bytes-per-event, e.g. 2*32*185*388> -f <binary-file-name>' +\
           '\n  %prog -b 40 -f data.bin' +\
           '\n  %prog -b 2296960 -f data-cxif5315-r129-cspad-dark.bin' +\
           '\n  %prog -b 1048576 -f data-amo86615-r197-pnccd-calib-spi.bin'

#------------------------------

def input_option_parser() :

    dsn_def  = None # 'exp=cxitut13:run=10' 
    exp_def  = 'cxitut13' 
    src_def  = 'CxiDs1.0:Cspad.0' 
    run_def  = 10
    nev_def  = 1000000
    nsk_def  = 0
    clb_def  = False
    fnm_def  = 'data.bin' 
    bsz_def  = 0
    vrb_def  = True
 
    h_dsn    = 'psana dataset name, e.g., exp=cxitut13:run=10, default = %s' % dsn_def
    h_exp    = 'experiment, e.g., cxi43210, default = %s' % exp_def
    h_run    = 'run number, default = %d' % run_def
    h_src    = 'data source name, e.g., cspad, default = %s' % src_def
    h_nev    = 'number of events to read from file, default = %d' % nev_def
    h_nsk    = 'number of events to skip before read, default = %d' % nsk_def
    h_clb    = 'save calib/raw data, default = %s' % str(clb_def)
    h_fnm    = 'binary-file name, default = %s' % fnm_def
    h_bsz    = 'buffer size - number of int16 (2-byte) words per dataset to read, default = %d' % bsz_def
    h_vrb    = 'verbosity, default = %s' % str(vrb_def)
 
    parser = OptionParser(description='Command line parameters', usage ='usage: %prog <opts>' + usage())
    parser.add_option('-d', '--dsn', default=dsn_def, action='store', type='string', help=h_dsn)
    parser.add_option('-e', '--exp', default=exp_def, action='store', type='string', help=h_exp)
    parser.add_option('-r', '--run', default=run_def, action='store', type='int',    help=h_run)
    parser.add_option('-s', '--src', default=src_def, action='store', type='string', help=h_src)
    parser.add_option('-n', '--nev', default=nev_def, action='store', type='int',    help=h_nev)
    parser.add_option('-m', '--nsk', default=nsk_def, action='store', type='int',    help=h_nsk)
    parser.add_option('-c', '--clb', default=clb_def, action='store_true',           help=h_clb)
    parser.add_option('-f', '--fnm', default=fnm_def, action='store', type='string', help=h_fnm)
    parser.add_option('-b', '--bsz', default=bsz_def, action='store', type='int',    help=h_bsz)
    parser.add_option('-v', '--vrb', default=vrb_def, action='store_false',          help=h_vrb)
  
    return parser #, parser.parse_args()

#------------------------------

if __name__ == "__main__" :

    parser = input_option_parser()

    if len(sys.argv) == 1 :
        parser.print_help()
        proc_name = os.path.basename(sys.argv[0])
        msg = '\nWARNING: run this command with parameters, e.g.: %s -h' % proc_name
        sys.exit ('%s\nEnd of %s' % (msg, proc_name))

    do_work(parser)

    sys.exit(0)

#------------------------------

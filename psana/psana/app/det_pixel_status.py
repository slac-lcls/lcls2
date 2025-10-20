#!/usr/bin/env python

DESCRIPTION = 'Evaluates pixel status of any detector/segment using raw dark/light data'

import sys
from psana.detector.dir_root import DIR_REPO as DIR_REPO_STATUS
from psana.detector.UtilsLogging import STR_LEVEL_NAMES  #logging

SCRNAME = sys.argv[0].split('/')[-1]

USAGE ='\n%s works with 2-d raw arrays ONLY! For 3-d the segment index --segind needs to be specified' % SCRNAME\
      +'\n%s -k <dataset> -d <detname> ...' % SCRNAME\
      +'\nTEST 1:  %s -k exp=mfx100848724,run=51 -d epix100_0 -F 1 -r 1000 -p1 -s' % SCRNAME\
      +'\n Feature # stands for'\
      +'\n   1: mean intensity of frames in good range'\
      +'\n   2: dark mean in good range'\
      +'\n   3: dark rms in good range'\
      +'\n   4: NOT-IMPLEMENTED for gain stage indicator values'\
      +'\n   5: NOT-IMPLEMENTED for gain stage indicator values'\
      +'\n   6: light average SNR of pixels over time'\
      +'\n  11: light intensity max-peds in good range - should be processed separately from all other features on entire/large set of events.'\
      +'\n'\
      +'\n\nHelp:  %s -h\n' % SCRNAME
#      +'\nTEST 1:  %s -k exp=rixx1003721,run=200,intg_det=epixhr -d epixhr -F 1,2,3' % SCRNAME\
#      +'\nEx.1:  %s -k exp=xpplw3319,run=287 -d XppGon.0:Epix100a.3 -F 1,2,3 -n 1000   # dark  processing for Feat.1 & 2 & 3' % SCRNAME\
#      +'\nEx.2:  %s -k exp=xpplw3319,run=293 -d XppGon.0:Epix100a.3 -F 1,6 -n 1000     # light processing for Feat.1 & 6' % SCRNAME\
#      +'\nEx.3:  %s -k exp=xpplw3319,run=293 -d XppGon.0:Epix100a.3 -F 11, -n 1000000  # light processing for Feat.11' % SCRNAME\


def argument_parser():
    from argparse import ArgumentParser

    d_dirrepo  = DIR_REPO_STATUS
#    d_dskwargs = 'exp=rixx1003721,run=200,intg_det=epixhr'  # 'exp=xpplw3319,run=293'
    d_dskwargs = 'exp=mfx100848724,run=51'
    d_detname  = 'epix100_0'
    d_nrecs    = 100
    d_events   = 1000
    d_evskip   = 0
    d_stepnum  = None
    d_stepmax  = None
    d_steps    = None  # '5,6,7'
    d_slice    = None  # '0:,0:'
    d_shwind   = '15,15'
    d_snrmax   = 8.0
    d_evcode   = None
    d_segind   = None
    d_gmode    = None
    d_loglevel = 'INFO'
    d_dirmode  = 0o2775
    d_filemode = 0o664
    d_group    = 'ps-users'
    d_databits = None # (1<<14)-1 # 0o37777=0x3fff=16383 -14 bit # 0o177777 -16bit
    d_gainbits = 0
    d_ctype    = 'status_data'
    d_features = '1,2,3'
    d_plotim   = 0
    d_version  = 'V2025-10-06'
    d_reset    = False
    #d_fname    = '/lscratch/dubrovin/tmp/EventLoopStatus-data_block-mfx101332224-r0204-jungfrau-seg003-step00-nrecs0100.npz'

    h_dskwargs = '(str) DataSource parameters, default = %s' % d_dskwargs
    h_detname  = '(str) detector/detname name, default = %s' % d_detname
    h_events   = '(int) maximal number of events total (in runs, steps), default = %d' % d_events
    h_evskip   = '(int) number of events to skip in the beginning of each step, default = %d' % d_evskip
    h_stepnum  = '(int) step number (in total for many runs counting from 0) to process or None for all steps, default = %s' % str(d_stepnum)
    h_stepmax  = '(int) maximal number of steps (total) to process or None for all steps, default = %s' % str(d_stepmax)
    h_steps    = '(str) list of step numbers (in total for many runs counting from 0) to process or None for all, default = %s' % str(d_steps)
    h_evcode   = '(str) comma separated event codes for selection as OR combination; any negative '\
                 'code inverts selection, default = %s' % str(d_evcode)
    h_segind   = '(int) segment index for multi-panel detectors with raw.shape.ndim>2, default = %s' % str(d_segind)
    h_gmode    = '(str) gain mode name-suffix for multi-gain detectors with raw.shape.ndim>3, ex: AHL-H, default = %s' % str(d_gmode)
    h_nrecs    = '(int) number of records to collect data, default = %s' % str(d_nrecs)
    h_dirrepo  = '(str) repository for calibration results, default = %s' % d_dirrepo
    h_loglevel = '(str) logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_loglevel)
    h_dirmode  = '(int) mode for all mkdir, default = %s' % oct(d_dirmode)
    h_filemode = '(int) mode for all saved files, default = %s' % oct(d_filemode)
    h_group    = '(str) group ownership for all files, default = %s' % d_group
    h_slice    = '(str) FOR DEBUGGING ONLY (str) slice of the panel image 2-k array selected for plots and pixel status,'\
                 ' ex. "0:144,0:192", default = %s' % d_slice
    h_shwind   = '(str) window shape for feature 6 fitting to plane, ex. "15,15", default = %s' % d_shwind
    h_snrmax   = '(int) width of the good region in terms on number of spread/rms, default = %s' % d_snrmax
    h_databits = '(int) data bits in ADC for code of intensity, default = %s' % (oct(d_databits) if d_databits is not None else\
                                                                                 'None - defined from det.raw._data_bit_mask')
    h_gainbits = '(int) gain mode switch bits in ADC, default = %s' % oct(d_gainbits)
    h_ctype    = '(str) type of calibration constants to save, default = %s' % d_ctype
    h_features = '(str) comma-separated list of Features [JAC-2022, Sadri, Automatic bad pixel mask maker...]'\
                 ' from 1 to 6 to evaluate bad pixels, default = %s' % d_features
    h_plotim   = 'plot image/s of pedestals, default = %s' % str(d_plotim)
    h_version  = 'constants version, default = %s' % str(d_version)
    h_reset    = '(bool) usereset tmp file to save/load data block, default = %s' % d_reset
    #h_fname    = '(str) input file name with accumulated data block and a few other parameters, default = %s' % d_fname

    parser = ArgumentParser(description=DESCRIPTION, usage=USAGE)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str,   help=h_detname)
    parser.add_argument('-n', '--events',   default=d_events,   type=int,   help=h_events)
    parser.add_argument('-m', '--evskip',   default=d_evskip,   type=int,   help=h_evskip)
    parser.add_argument('-N', '--stepnum',  default=d_stepnum,  type=int,   help=h_stepnum)
    parser.add_argument('-M', '--stepmax',  default=d_stepmax,  type=int,   help=h_stepmax)
    parser.add_argument('--steps',          default=d_steps,    type=int,   help=h_steps)
    parser.add_argument('-c', '--evcode',   default=d_evcode,   type=str,   help=h_evcode)
    parser.add_argument('-i', '--segind',   default=d_segind,   type=int,   help=h_segind)
    parser.add_argument('-S', '--slice',    default=d_slice,    type=str,   help=h_slice)
    parser.add_argument('-r', '--nrecs',    default=d_nrecs,    type=int,   help=h_nrecs)
    parser.add_argument('-w', '--shwind',   default=d_shwind,   type=str,   help=h_shwind)
    parser.add_argument('-R', '--snrmax',   default=d_snrmax,   type=float, help=h_snrmax)
    parser.add_argument('-t', '--ctype',    default=d_ctype,    type=str,   help=h_ctype)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str,   help=h_loglevel)
    parser.add_argument('-o', '--dirrepo',  default=d_dirrepo,  type=str,   help=h_dirrepo)
    parser.add_argument('-F', '--features', default=d_features, type=str,   help=h_features)
    parser.add_argument('--dirmode',        default=d_dirmode,  type=int,   help=h_dirmode)
    parser.add_argument('--filemode',       default=d_filemode, type=int,   help=h_filemode)
    parser.add_argument('--group',          default=d_group,    type=str,   help=h_group)
    parser.add_argument('--databits',       default=d_databits, type=int,   help=h_databits)
    parser.add_argument('--gainbits',       default=d_gainbits, type=int,   help=h_gainbits)
    parser.add_argument('--gmode',          default=d_gmode,    type=str,   help=h_gmode)
    parser.add_argument('-p', '--plotim',   default=d_plotim,   type=int,   help=h_plotim)
    parser.add_argument('-v', '--version',  default=d_version,  type=str,   help=h_version)
    parser.add_argument('-s', '--reset',    action='store_true',            help=h_reset)
    #parser.add_argument('-f', '--fname',    default=d_fname,    type=str,   help=h_fname)

    return parser


def do_main():

    if len(sys.argv)<2:
        sys.exit('%s\n%s\n%s\nEXIT - MISSING PARAMETERS' % (40*'_', DESCRIPTION, USAGE))

    parser = argument_parser()
    args = parser.parse_args() # NameSpace
    kwargs = vars(args)        # dict

    import psana.detector.Utils as ut #info_dict, info_command_line, info_namespace, info_parser_arguments, str_tstamp
    #print(ut.info_parser_arguments(parser))

    from psana.detector.UtilsPixelStatus import det_pixel_status
    det_pixel_status(parser)

    sys.exit(0)

if __name__ == "__main__":
    do_main()

# EOF



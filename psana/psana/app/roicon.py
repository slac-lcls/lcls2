#!/usr/bin/env python

"""Constructor/converter of the image-ROI mask to data-shaped ndarray mask using geometry calib file.

This software was developed for the lcls2 project.
If you use all or part of it, please give an appropriate acknowledgment.

2023-02-10 adopted from lcls1 CalibManager/app/roicon by Mikhail Dubrovin
"""

from psana.detector.UtilsLogging import sys, logging, DICT_NAME_TO_LEVEL, init_stream_handler

logger = logging.getLogger(__name__)
SCRNAME = sys.argv[0].rsplit('/')[-1]
DIR_REPO = './work'
TESTNDA = '<ndarray-shaped-as-data-fname>'
TGFNAME = '<geometry-fname>'
TGFNAME1 = '/cds/group/psdm/detector/data_test/geometry/geo-epix10ka2m-16-segment.data'

USAGE = '\n'\
      + '\n1) Construct 2-d image (or mask-of-segments) from ndarray with image shaped as data using appropriate geometry file'\
      + '\n         %s 1 -g <geometry-file> [-a (input)%s] [-i <image-(output)file>] [-c <control-bitword>]' % (SCRNAME, TESTNDA)\
      + '\n  ex1:   %s 1 -g %s' % (SCRNAME, TGFNAME)\
      + '\n  ex2:   %s 1 -g %s -a %s' % (SCRNAME, TGFNAME, TESTNDA)\
      + '\n  test:  %s 1 -g %s -i test-2d-mask.npy -t' % (SCRNAME, TGFNAME1)\
      + '\n\n2) (TBD) Create ROI mask using mask editor "med" (DO NOT FORGET to save mask in file!)'\
      + '\n         %s 2 [-i <image-(input)file>] [-m <roi-mask-(output)file>]' % SCRNAME\
      + '\n  ex1,2: %s 2' % SCRNAME\
      + '\n  ex3:   %s 2 -i image.npy -m roi-mask.npy' % SCRNAME\
      + '\n\n3) Convert ROI mask to ndarray with mask shaped as data'\
      + '\n         %s 3 -g <geometry-file> [-m <roi-mask-(input)file>] [-n ndarray-with-mask-(output)-file] [-c <control-bitword>]' % SCRNAME\
      + '\n  ex1,2: %s 3 -g %s' % (SCRNAME, TGFNAME)\
      + '\n  test:  %s 3 -g %s -m test-2d-mask.npy -n test-3d-mask.npy' % (SCRNAME, TGFNAME1)


def argument_parser():
    import argparse

    d_proc   = 1
    d_gfname = TGFNAME
    d_afname = None
    d_ifname = 'mask-img.txt'
    d_mfname = 'mask-roi.txt'
    d_nfname = 'mask-nda.txt'
    d_cbits  = 0xffff
    d_verb   = False
    d_dotest = False
    d_save   = False
    d_dirrepo = DIR_REPO
    d_logmode = 'INFO'
    d_kwargs = '{}'

    h_proc   = 'process number: 1-construct image, 2-run mask editor on image, 3-convert image mask to ndarray; default = %s' % d_proc
    h_gfname = 'geometry file name, default = %s' % d_gfname
    h_afname = 'input ndarray file name, default = %s' % d_afname
    h_ifname = 'image file name, default = %s' % d_ifname
    h_mfname = 'ROI mask file name, default = %s' % d_mfname
    h_nfname = 'ndarray mask file name, default = %s' % d_nfname
    h_cbits  = 'mask control bits, =0-none, +1-edges, +2-middle, etc..., default = %d' % d_cbits
    h_verb   = 'verbosity, default = %s' % str(d_verb)
    h_dotest = 'add a couple of rings to the 2-d mask for test purpose, default = %s' % str(d_dotest)
    h_save   = 'save plots, default = %s' % str(d_save)
    h_dirrepo = 'repository for logs and output files, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (' '.join(DICT_NAME_TO_LEVEL.keys()), d_logmode)
    h_kwargs = 'str python code evaluated to dict and passed to geo.get_pixel_coord_indexes(**kwargs), default = %s' % d_kwargs

    parser = argparse.ArgumentParser(description='Conversion between 2-d and 3-d masks.', usage=USAGE)
    #parser.add_argument('args', nargs='*', default=d_proc, help=h_proc)
    parser.add_argument('args', nargs='?', default=d_proc, help=h_proc) # '?' - no list, just a single str parameter
    parser.add_argument('-g', '--gfname', default=d_gfname, type=str, help=h_gfname)
    parser.add_argument('-a', '--afname', default=d_afname, type=str, help=h_afname)
    parser.add_argument('-i', '--ifname', default=d_ifname, type=str, help=h_ifname)
    parser.add_argument('-m', '--mfname', default=d_mfname, type=str, help=h_mfname)
    parser.add_argument('-n', '--nfname', default=d_nfname, type=str, help=h_nfname)
    parser.add_argument('-c', '--cbits',  default=d_cbits,  type=int, help=h_cbits)
    parser.add_argument('-v', '--verb',   action='store_true', help=h_verb)
    parser.add_argument('-t', '--dotest',  action='store_true', help=h_dotest)
    parser.add_argument('-S', '--save',    action='store_true', help=h_save)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo, type=str, help=h_dirrepo)
    parser.add_argument('-L', '--logmode', default=d_logmode, type=str, help=h_logmode)
    parser.add_argument('-k', '--kwargs', default=d_kwargs, type=str, help=h_kwargs)
    return parser


def run_parser_do_main():

    if len(sys.argv)==1:
        sys.exit('Try command: %s -h' % SCRNAME)

    parser = argument_parser()
    nspace = parser.parse_args()
    kwargs = vars(nspace) # Namespace > dict
    defs = vars(parser.parse_args([])) # dict of defaults only

    init_stream_handler(loglevel=nspace.logmode)

    s = '\ncommand: %s' % ' '.join(sys.argv)\
      + '\nkwargs : %s' % str(kwargs)\
      + '\nnspace : %s' % str(nspace)\
      + '\nargs   : %s' % str(nspace.args)\
      + '\ndefs   : %s' % str(defs)

    logger.info(s)

    from psana.detector.utils_roicon import do_main
    do_main(parser)

    sys.exit('End of %s' % SCRNAME)

#if __name__ == "__main__":
run_parser_do_main() # to run it as __main__ from setup.py

# EOF

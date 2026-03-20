#!/usr/bin/env python

from time import time
t0_sec_tot = time()

import sys
#logger = logging.getLogger(__name__)
from psana.detector.dir_root import DIR_REPO_JUNGFRAU
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES

SCRNAME = sys.argv[0].rsplit('/')[-1]

M14 = 0x3fff # 0o37777, 16383, 14-bit of data mask, 2 bits for gain mode switch

USAGE = 'Usage:'\
      + f'\n  {SCRNAME} -k <\"str-of-datasource-kwargs\"> -d <detector>'\
      + '\n     [-o <output-result-directory>] [-L <logging-mode>] [other-kwargs]'\
      + '\nTests:'\
      + '\n  datinfo -k exp=mfx100848724,run=49 -d jungfrau ### TEST DATA'\
      + f'\n  {SCRNAME} -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 50 --nrecs1 50 ### STAGE 1 ONLY'\
      + f'\n  {SCRNAME} -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0 ### STAGE 2 ONLY'\
      + f'\n  mpirun --mca osc ^ucx -n 5 {SCRNAME} -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 0 ### STAGE 2 ONLY WITH MPIRUN'\
      + f'\n  {SCRNAME} -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 1000 --nrecs1 50 --wrapper --submit ### RUN WRAPPER WITHOUT EXECUTING COMMANDS'\
      + f'\nHELP:\n  {SCRNAME} -h'
#      + '\n'\


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs= 'exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc' # None
    d_detname = 'jungfrau' #  None
    d_nrecs   = 1000  # number of records to collect and process
    d_nrecs1  = 50    # number of records to process at 1st stage
    d_dirrepo = DIR_REPO_JUNGFRAU # './work'
    d_logmode = 'INFO'
    d_errskip = True
    d_stepnum = None
    d_stepmax = 3
    d_evskip  = 0       # number of events to skip in the beginning of each step
    d_events  = 10000   # max_events in DataSource(max_events=events,...)
    d_dirmode = 0o2775
    d_filemode= 0o664
    d_group   = 'ps-users'
    d_int_lo  = 1       # lowest  intensity accepted for dark evaluation
    d_int_hi  = M14-1   # highest intensity accepted for dark evaluation, ex: 16000
    d_intnlo  = 6.0     # intensity ditribution number-of-sigmas low
    d_intnhi  = 6.0     # intensity ditribution number-of-sigmas high
    d_rms_lo  = 0.001   # rms distribution low
    d_rms_hi  = M14-1   # rms distribution high, ex: 16000
    d_rmsnlo  = 6.0     # rms distribution number-of-sigmas low
    d_rmsnhi  = 6.0     # rms distribution number-of-sigmas high
    d_fraclm  = 0.1     # allowed fraction limit
    d_fraclo  = 0.05    # fraction of statistics [0,1] below low limit
    d_frachi  = 0.95    # fraction of statistics [0,1] below high limit
    d_version = 'V2026-03-05'
    d_datbits = M14     # 14-bits, 2 bits for gain mode switch
    d_ctdepl  = 'psr'   # for constants from dark, 'psrnx'
    d_deploy  = False
    d_save    = False
    d_plotim  = 0
    d_segind  = None
    d_igmode  = None
    d_wrapper = False
    d_submit  = False
    d_stages  = 7
    d_nranks  = 1
    d_nnodes  = 1

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_nrecs   = 'number of records to calibrate pedestals, default = %s' % str(d_nrecs)
    h_detname = 'detector name, default = %s' % d_detname
    h_nrecs1  = 'number of records to process at 1st stage, default = %s' % str(d_nrecs1)
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_errskip = 'flag to skip errors and keep processing, stop otherwise, default = %s' % d_errskip
    h_stepnum = 'step number to process or None for all steps, default = %s' % str(d_stepnum)
    h_stepmax = 'maximum number of steps to process, default = %s' % str(d_stepmax)
    h_evskip  = 'number of events to skip in the beginning of each step, default = %s' % str(d_evskip)
    h_events  = 'total number of events to read from xtc2 file, DataSource(..., max_events=events, ...), default = %s' % str(d_events)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_int_lo  = 'lowest  intensity accepted for dark evaluation, default = %d' % d_int_lo
    h_int_hi  = 'highest intensity accepted for dark evaluation, for None derived from data_bit_mask, default = %s' % d_int_hi
    h_intnlo  = 'intensity ditribution number-of-sigmas low, default = %f' % d_intnlo
    h_intnhi  = 'intensity ditribution number-of-sigmas high, default = %f' % d_intnhi
    h_rms_lo  = 'rms ditribution low, default = %f' % d_rms_lo
    h_rms_hi  = 'rms ditribution high, for None derived from data_bit_mask, default = %s' % d_rms_hi
    h_rmsnlo  = 'rms ditribution number-of-sigmas low, default = %f' % d_rmsnlo
    h_rmsnhi  = 'rms ditribution number-of-sigmas high, default = %f' % d_rmsnhi
    h_fraclm  = 'fraction of statistics [0,1] below low or above high gate limit to assign pixel bad status, default = %f' % d_fraclm
    h_fraclo  = 'fraction of statistics [0,1] below low  limit of the gate, default = %f' % d_fraclo
    h_frachi  = 'fraction of statistics [0,1] above high limit of the gate, default = %f' % d_frachi
    h_version = 'constants version, default = %s' % str(d_version)
    h_datbits = 'data bits, e.g. 0x3fff is 14-bit mask for jungfrau with 2 bits for gain modes, default = %s' % hex(d_datbits)
    h_save    = 'save constants in repository, default = %s' % d_save
    h_ctdepl    = '(str) keyword for deployment: "p"-pedestals, "r"-rms, "s"-status, "x" - max, "n" - min, default = %s' % d_ctdepl
    h_deploy  = 'deploy constants to the calibration DB, default = %s' % d_deploy
    h_plotim  = 'plot image/s of pedestals, default = %s' % str(d_plotim)
    h_segind  = 'segment index in det.raw.raw array to process, default = %s' % str(d_segind)
    h_igmode  = 'gainmode index FOR DEBUGGING, default = %s' % str(d_igmode)
    h_wrapper = 'FOR WRAPPER: directly run wrapper jungfrau_dark_proc_mpi.sh in stead of python script, default = %s' % d_wrapper
    h_submit  = 'FOR WRAPPER: submit commands for execution, otherwise show what script is doing for debugging, default = %s' % d_submit
    h_stages  = 'FOR WRAPPER: bitword 001/010/100 for stages 1/2/3, respectively, or any bit combination, default = %d' % d_stages
    h_nranks  = 'FOR WRAPPER: passed to sbatch --ntasks-per-node=NRANKS, default = %s' % d_nranks
    h_nnodes  = 'FOR WRAPPER: number of nodes FOR NOW works for 1 ONLY, default = %s' % d_nnodes

    parser = ArgumentParser(usage=USAGE, description='Proceses dark run xtc2 data for jungfrau')
    parser.add_argument('-k', '--dskwargs',default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname', default=d_detname,    type=str,   help=h_detname)
    parser.add_argument('-n', '--nrecs',   default=d_nrecs,      type=int,   help=h_nrecs)
    parser.add_argument('--nrecs1',        default=d_nrecs1,     type=int,   help=h_nrecs1)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-L', '--logmode', default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('-E', '--errskip', action='store_false',             help=h_errskip)
    parser.add_argument('--stepnum',       default=d_stepnum,    type=int,   help=h_stepnum)
    parser.add_argument('--stepmax',       default=d_stepmax,    type=int,   help=h_stepmax)
    parser.add_argument('--evskip',        default=d_evskip,     type=int,   help=h_evskip)
    parser.add_argument('--events',        default=d_events,     type=int,   help=h_events)
    parser.add_argument('--dirmode',       default=d_dirmode,    type=int,   help=h_dirmode)
    parser.add_argument('--filemode',      default=d_filemode,   type=int,   help=h_filemode)
    parser.add_argument('--int_lo',        default=d_int_lo,     type=int,   help=h_int_lo)
    parser.add_argument('--int_hi',        default=d_int_hi,     type=int,   help=h_int_hi)
    parser.add_argument('--intnlo',        default=d_intnlo,     type=float, help=h_intnlo)
    parser.add_argument('--intnhi',        default=d_intnhi,     type=float, help=h_intnhi)
    parser.add_argument('--rms_lo',        default=d_rms_lo,     type=float, help=h_rms_lo)
    parser.add_argument('--rms_hi',        default=d_rms_hi,     type=float, help=h_rms_hi)
    parser.add_argument('--rmsnlo',        default=d_rmsnlo,     type=float, help=h_rmsnlo)
    parser.add_argument('--rmsnhi',        default=d_rmsnhi,     type=float, help=h_rmsnhi)
    parser.add_argument('--fraclm',        default=d_fraclm,     type=float, help=h_fraclm)
    parser.add_argument('--fraclo',        default=d_fraclo,     type=float, help=h_fraclo)
    parser.add_argument('--frachi',        default=d_frachi,     type=float, help=h_frachi)
    parser.add_argument('-v', '--version', default=d_version,    type=str,   help=h_version)
    parser.add_argument('--datbits',       default=d_datbits,    type=int,   help=h_datbits)
    parser.add_argument('-S', '--save',    action='store_true',              help=h_save)
    parser.add_argument('-D', '--deploy',  action='store_true',              help=h_deploy)
    parser.add_argument('-p', '--ctdepl',  default=d_ctdepl,     type=str,   help=h_ctdepl)
    parser.add_argument('-i', '--plotim',  default=d_plotim,     type=int,   help=h_plotim)
    parser.add_argument('-I', '--segind',  default=d_segind,     type=int,   help=h_segind)
    parser.add_argument('-G', '--igmode',  default=d_igmode,     type=int,   help=h_igmode)
    parser.add_argument('--wrapper',       action='store_true',              help=h_wrapper)
    parser.add_argument('--submit',        action='store_true',              help=h_submit)
    parser.add_argument('--stages',        default=d_stages,     type=int,   help=h_stages)
    parser.add_argument('--nranks',        default=d_nranks,     type=int,   help=h_nranks)
    parser.add_argument('--nnodes',        default=d_nnodes,     type=int,   help=h_nnodes)

    return parser


def do_main():
    from time import time
    t0_sec = time()

    parser = argument_parser()
    args = parser.parse_args()
    kwa = vars(args)

    if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.detname  is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'
#    assert args.stepnum  is not None, 'WARNING: option "--stepnum <stepnum>" MUST be specified.'

#    if use_mpi: from psana.detector.UtilsJungfrauCalibMPI import jungfrau_dark_proc
#    else:       from psana.detector.UtilsJungfrauCalib    import jungfrau_dark_proc

    if args.wrapper:
        import os
        scr_dir = os.path.dirname(os.path.abspath(__file__))
        scr_name = f'{scr_dir}/jungfrau_dark_proc_wrapper.sh'
        cmd = f'{scr_name} -k {args.dskwargs} -d {args.detname} --nrecs {args.nrecs} --nrecs1 {args.nrecs1} --dirrepo {args.dirrepo} --logmode {args.logmode}'\
            + f' --datbits {args.datbits} --int_lo {args.int_lo} --int_hi {args.int_hi} --fraclo {args.fraclo} --frachi {args.frachi}'\
            + f' --stages {args.stages} --nranks {args.nranks} --nnodes {args.nnodes} ' # for debugging
        if args.submit: cmd += ' --submit'
        print(f'RUN SHELL SCRIPT-WRAPPER FOR A SEQUENCE OF COMMANDS:\n{cmd}')
        os.system(cmd)
    else:
        from psana.detector.UtilsJungfrauCalibMPI import jungfrau_dark_proc
        jungfrau_dark_proc(parser)

    print('%s %s TOTAL TIME %.3f sec' % (SCRNAME, 'shell wrapper' if args.wrapper else 'script', time() - t0_sec_tot))

if __name__ == "__main__":
    do_main()
    sys.exit(0)

# EOF

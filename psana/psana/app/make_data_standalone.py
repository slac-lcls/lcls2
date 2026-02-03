#!/usr/bin/env python
"""
2025-12-18 - development of script psana/psana/pycalgos/test_make_data_for_standalone.py,
             add regular psana data access parameters
"""
import sys
from psana.detector.NDArrUtils import info_ndarr
import psana.detector.Utils as ut
import psana.detector.utils_psana as up
import logging
#logger = logging.getLogger
SCRNAME = sys.argv[0].rsplit('/')[-1]

def ds_run_det(**kwargs):
    """make data file with det.raw per event only for standalone tests"""
    from psana import DataSource
    str_dskwargs = kwargs.get('dskwargs', 'exp=mfx100848724,run=51')
    detname      = kwargs.get('detname', 'jungfrau')
    events       = kwargs.get('events', 100)
    try:
      dskwargs = up.datasource_kwargs_from_string(str_dskwargs)
      dskwargs['max_events'] = events
      dskwargs['batch_size'] = 1
      print('ds_run_det with **dskwargs=%s' % str(dskwargs))
      ds = DataSource(**dskwargs)
    except:
      print('DataSource IS NOT AVAILABLE')
      sys.exit()
    run = next(ds.runs())
    det = run.Detector(detname)
    return ds, run, det, dskwargs


def raw_in_event_loop(**kwargs):
    """2025-12-02 prints raw in the event loop"""
    print('\ntest raw_in_event_loop\n')
    events = kwargs.get('events', 10)
    ds, run, det, dskwargs = ds_run_det(**kwargs)
    for nevt,evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        if nevt>events: break
        print(info_ndarr(raw, 'evt:%03d raw' % nevt, last=10, vfmt='%d'))


def make_data(**kwargs):
    """generate data file for standalone tests"""
    print('\nin make_data\n')
    from psana.detector.Utils import selected_record
    ds, run, det, dskwargs = ds_run_det(**kwargs)

    exp     = dskwargs.get('exp', None)
    runnum  = dskwargs.get('run', 0)
    detname = kwargs.get('detname', None)
    events  = kwargs.get('events', 10)
    fname_def = 'raw_data_%s_r%04d_e%06d_%s.dat' % (exp, runnum, events, detname)
    fnsuff  = kwargs.get('fname', fname_def)
    fname   = tmp_filename(fname=fnsuff)

    outfile = open(fname,'w')
    for nevt,evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        raw.tofile(outfile)
        if selected_record(nevt, events=events):
           print(info_ndarr(raw, 'evt:%05d raw' % nevt, last=10, vfmt='%d'))
        if nevt>events: break
    outfile.close()
    print('raw data saved in file: %s' % fname)


def tmp_filename(fname=None, suffix='_calib_constants.dat'):
   """returns file name in
      /lscratch/<username>/tmp/fname   if fname is not None or
      /lscratch/<username>/tmp/<random-str>suffix
   """
   import os
   import tempfile
   tmp_file = tempfile.NamedTemporaryFile(mode='r+b', suffix=suffix)
   return tmp_file.name if fname is None else\
          os.path.join(os.path.dirname(tmp_file.name), fname)


def make_calibcons(**kwargs):
    """2025-12-01 save calib constants for calib of version cversion 1/2/3"""

    import psana.detector.UtilsJungfrau as uj
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.INFO) #logging.DEBUG)

    ds, run, det, dskwargs = ds_run_det()
    evt = next(run.events())
    cversion = kwargs.get('cversion', 3)
    print('\nin make_calibcons cversion:%d\n' % cversion)
    exp     = dskwargs.get('exp', None)
    runnum  = dskwargs.get('run', 0)
    detname = kwargs.get('detname', None)
    fname_def = 'calibcons_v%d_%s_r%04d_%s.dat' % (cversion, exp, runnum, detname)
    fnsuff = kwargs.get('fname', fname_def)
    fname=tmp_filename(fname=fnsuff)
    kwa = {'cversion': cversion,}
    odc = uj.DetCache(det.raw, evt, **kwa) # cache.add_detcache(det_raw, evt, **kwa)
    #print(det.raw._info_calibconst()) # is called in AreaDetector

    cc = odc.ccons
    print(info_ndarr(cc, 'ccons for calib', last=10, vfmt='%0.3f'))
    outfile = open(fname,'w')
    cc.tofile(outfile)
    outfile.close()
    print('calib constants saved in file: %s' % fname)


def argument_parser():
    from argparse import ArgumentParser
    d_tname    = '0'
    d_dskwargs = 'exp=mfx100848724,run=51'  # None
    d_detname  = 'jungfrau' # None
    d_events   = 10
    d_loglevel = 'INFO' # 'DEBUG'
    d_plot_img = 0
    d_cversion = 3

    h_tname    = '(str) test name, usually numeric, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_events   = 'number of events to process, default = %d' % d_events
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    h_plot_img = 'image bitword to plot images, default = %d' % d_plot_img
    h_cversion = 'version of JF calib constants 1/2/3: shape=(npix,2,4)/(8,npix)/(4,npix,2), default = %d' % d_cversion

    parser = ArgumentParser(description='%s saves raw data for specified detector in file' % SCRNAME, usage=usage())
    parser.add_argument('-t', '--tname',    default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-n', '--events',   default=d_events,   type=int, help=h_events)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-p', '--plot_img', default=d_plot_img, type=int, help=h_plot_img)
    parser.add_argument('-v', '--cversion', default=d_cversion, type=int, help=h_cversion)
    #parser.add_argument('-f', '--fname',    default=d_cversion, type=int, help=h_cversion)
    return parser


def usage():
    import inspect
    return '\n  example: %s -t0 -k exp=mfx100848724,run=51 -d jungfrau' % SCRNAME \
        +'\n\n  %s -t <tname> [other kwargs]\n' % sys.argv[0].split('/')[-1]\
    + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "tname ==" in s or "tnum in" in s])


def selector():
    if len(sys.argv) < 2:
        print(usage())
        sys.exit('EXIT due to MISSING PARAMETERS')

    parser = argument_parser()
    args = parser.parse_args()
    kwargs = vars(args)

    #print('parser.parse_args()', args)
    print(ut.info_parser_arguments(parser, title='parser parameters:'))

    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

    tname = args.tname  # sys.argv[1] if len(sys.argv)>1 else '0'
    tnum = int(tname)

    if   tname == '0': raw_in_event_loop(**kwargs)
    elif tname == '1': make_calibcons(**kwargs)
    elif tname == '2': make_data(**kwargs)
    elif tname == '3': # make both, calibconstants and data
        make_calibcons(**kwargs)
        make_data(**kwargs)
    else: exit('\nTEST "%s" IS NOT IMPLEMENTED' % tname)


if __name__ == "__main__":
    selector()

# EOF

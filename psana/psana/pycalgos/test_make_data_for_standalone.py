#!/usr/bin/env python
"""
   ./test_make_data_for_standalone.py 0      # test raw in event loop
   ./test_make_data_for_standalone.py 1      # save calib constants in tmp file
   ./test_make_data_for_standalone.py 2      # save raw data for  100 events in tmp file
   ./test_make_data_for_standalone.py 2 6500 # save raw data for 6500 events in tmp file
   ./test_make_data_for_standalone.py        # save both calib constants and data in tmp file
2025-12-05 - created
"""
import sys
from psana.detector.NDArrUtils import info_ndarr

def ds_run_det(exp='mfx100848724', runnum=51, detname='jungfrau'):
    """2025-12-01 created by Chris to generate data file for standalone test of calib"""
    from psana import DataSource
    print('ds_run_det for exp: %s runnum: %d detname: %s' % (exp, runnum, detname))
    try:
      ds = DataSource(exp=exp, run=runnum)
    except:
      print('DataSource IS NOT AVAILABLE')
      sys.exit()
    run = next(ds.runs())
    det = run.Detector(detname)
    return ds, run, det


def raw_in_event_loop(events=10):
    """2025-12-02 prints raw in the event loop"""
    print('\ntest raw_in_event_loop\n')
    ds, run, det = ds_run_det()
    for nevt,evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        if nevt>events: break
        print(info_ndarr(raw, 'evt:%03d raw' % nevt, last=10, vfmt='%d')) # is called in AreaDetector


def make_data(events=100, fname='/sdf/data/lcls/ds/xpp/xpptut15/scratch/cpo/cpojunk.dat'):
    """2025-12-01 created by Chris to generate data file for standalone test of calib"""
    print('\nin make_data\n')
    from psana.detector.Utils import selected_record
    ds, run, det = ds_run_det()
    outfile = open(fname,'w')
    for nevt,evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        raw.tofile(outfile)
        if selected_record(nevt, events=events):
           print(info_ndarr(raw, 'evt:%05d raw' % nevt, last=10, vfmt='%d')) # is called in AreaDetector
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


def make_calibcons(cversion=3, fname='calibcons_v3.dat'):
    """2025-12-01 savea calib constants for calib of version cversion 1/2/3"""
    print('\nin make_calibcons cversion:%d\n' % cversion)

    import psana.detector.UtilsJungfrau as uj
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.INFO) #logging.DEBUG)

    ds, run, det = ds_run_det()
    evt = next(run.events())
    kwa = {'cversion': cversion,}
    odc = uj.DetCache(det.raw, evt, **kwa) # cache.add_detcache(det_raw, evt, **kwa)
    #print(det.raw._info_calibconst()) # is called in AreaDetector

    cc = odc.ccons
    print(info_ndarr(cc, 'ccons for calib', last=10, vfmt='%0.3f'))       # shape:(4, 16777216, 2)
    outfile = open(fname,'w')
    cc.tofile(outfile)
    outfile.close()
    print('calib constants saved in file: %s' % fname)


if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv)>1 else '3'
    events = int(sys.argv[2]) if len(sys.argv)>2 else 100
    if   tname == '0': raw_in_event_loop(events=10)
    elif tname == '1': make_calibcons(cversion=3, fname=tmp_filename(fname='calibcons_v3.dat'))
    elif tname == '2': make_data(events=events,   fname=tmp_filename(fname='raw_data_mfx100848724_r051_e%06d.dat' % events))
    elif tname == '3': # make both calibconstants and data
        make_calibcons(cversion=3, fname=tmp_filename(fname='calibcons_v3.dat'))
        make_data(events=events,   fname=tmp_filename(fname='raw_data_mfx100848724_r051_e%06d.dat' % events))
    else: exit('\nTEST "%s" IS NOT IMPLEMENTED' % tname)

# EOF

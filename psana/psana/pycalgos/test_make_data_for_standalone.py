#!/usr/bin/env python

import sys
from psana.detector.NDArrUtils import info_ndarr


def ds_run_det():
    """2025-12-01 created by Chris to generate data file for standalone test of calib"""
    from psana import DataSource
    ds = DataSource(exp='mfx100848724', run=51)
    run = next(ds.runs())
    det = run.Detector('jungfrau')
    return ds, run, det


def raw_in_event_loop(events=10):
    """2025-12-02 prints raw in the event loop"""
    ds, run, det = ds_run_det()
    for nevt,evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        if nevt>events: break
        print(info_ndarr(raw, 'evt:%03d raw' % nevt, last=10, vfmt='%d')) # is called in AreaDetector


def make_data(ofname='/sdf/data/lcls/ds/xpp/xpptut15/scratch/cpo/cpojunk.dat'):
    """2025-12-01 created by Chris to generate data file for standalone test of calib"""
    ds, run, det = ds_run_det()
    outfile = open(ofname,'w')
    for nevt,evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        raw.tofile(outfile)
        if nevt>100: break
    outfile.close()


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


def make_calibcons(cversion=3, ofname='calibcons_v3.dat'):
    """2025-12-01 savea calib constants for calib of version cversion 1/2/3"""
    print('in make_calibcons cversion:%d' % cversion)

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
    fname = tmp_filename(fname=ofname, suffix='_calib_constants.dat')
    outfile = open(fname,'w')
    cc.tofile(outfile)
    outfile.close()
    print('saved in file: %s' % fname)


if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv)>1 else '1'
    if   tname == '0': raw_in_event_loop(events=10)
    elif tname == '1': make_calibcons(cversion=3, ofname='calibcons_v3.dat')
    elif tname == '2': make_data()
    else: exit('\nTEST "%s" IS NOT IMPLEMENTED'%tname)

# EOF

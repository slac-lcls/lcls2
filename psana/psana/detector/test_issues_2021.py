#!/usr/bin/env python
import logging
#logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s]: %(message)s', level=logging.INFO)

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]

def issue_2020_11_09():
    from psana import DataSource
    ds = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005.xtc2')

    orun = next(ds.runs())
    #for orun in ds.runs():
    det = orun.Detector('epix10k2M')
    for evnum,evt in enumerate(orun.events()):
        print('%s\nEvent %04d' % (50*'_',evnum))
        raw = det.raw.raw(evt)
        print('type(raw)',type(raw))
        #for segment,panel in raw.items():
        for panel in raw:
            print(panel.shape)
    print(50*'-')


def issue_2020_11_24():
    from psana import DataSource
    import numpy as np
    ds = DataSource(exp='tmolw0618',run=52)
    myrun = next(ds.runs())
    det = myrun.Detector('tmoopal')
    print(det.calibconst)


def issue_2020_12_02():
    from psana import DataSource
    print('DATA FILE IS AVAILABLE ON daq-det-drp01 ONLY')

    ds = DataSource(files='/u2/lcls2/tst/tstx00117/xtc/tstx00117-r0147-s000-c000.xtc2')

    #print('\nXXX dir(ds):', dir(ds))
    print('XXX ds.runnum: ', ds.runnum) # None
    print('XXX ds.exp   : ', ds.exp)    # None

    for i,run in enumerate(ds.runs()):
        print('\n=== run enum: %d', i)
        #print('\nXXX dir(run):', dir(run))
        print('XXX runnum      : ', run.runnum)
        print('XXX run.detnames: ', run.detnames) # {'epixquad'}
        print('XXX run.expt    : ', run.expt)     # amox27716
        print('XXX run.id      : ', run.id)       # 0
        #det = run.Detector('epixquad') 
    exit('TEST EXIT') 


def issue_2020_12_10():
    """
    O'Grady, Paul Christopher <cpo@slac.stanford.edu>
    Thu 12/10/2020 9:37 PM
    On Dec 10, 2020, at 10:47 AM, Dubrovin, Mikhail <dubrovin@slac.stanford.edu> wrote:

    >> Detector scan_raw_2_0_0 is implemented in detector/envstore.py
    >>but I can't get anything useful from this detector. Something is still missing.

    The script below works for me for the ued epix-scan.
    At the moment I?ve forgotten why we did it with two Detectors 
    instead of putting the information into one detector 
    or the Step object (like we do with the runinfo).
    Perhaps Mona/ChrisF/Matt (cc?d) can remind me post tag-up.
    We?ve had several long discussions about this and changed it a few times ...
    my apologies for forgetting.   This time I will capture what we say here:
    https://confluence.slac.stanford.edu/pages/viewpage.action?pageId=247694685
    chris
    """
    #print('DATA FILE IS AVAILABLE ON drp-ued-cmp001 ONLY')
    #fname = '/u2/pcds/pds/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2'

    from psana import DataSource
    ds = DataSource(exp='ueddaq02',run=28)
    #ds = DataSource(files='/reg/d/psdm/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2')
    #detname='epixquad'
    myrun = next(ds.runs())
    step_value = myrun.Detector('step_value')
    step_docstring = myrun.Detector('step_docstring')
    for nstep,step in enumerate(myrun.steps()):
        print('step:',nstep,step_value(step),step_docstring(step))
        for nevt,evt in enumerate(step.events()):
            if nevt==3: print('evt3:',nstep,step_value(evt),step_docstring(evt))


def issue_2020_12_16():
    """Chris' access to config does not work, Matts' works """
    from psana.pyalgos.generic.NDArrUtils import print_ndarr
    from psana import DataSource
    ds = DataSource(files='/cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2')
    detname = 'epixquad'

    for orun in ds.runs():
      print('run.runnum: %d detnames: %s expt: %s' % (orun.runnum, str(orun.detnames), orun.expt))

      det = orun.Detector(detname)
      print('det.raw._det_name: %s' % det.raw._det_name) # epixquad
      print('det.raw._dettype : %s' % det.raw._dettype)  # epix

      scfg = None
      for config in det._configs:
          if not detname in config.__dict__:
              print('Skipping config {:}'.format(config.__dict__))
              continue
          scfg = getattr(config,detname)

      for nstep, step in enumerate(orun.steps()):
          print('\n==== step:%02d' %nstep)

          print('DOES NOT WORK')
          for k,v in det.raw._seg_configs().items():
              cob = v.config
              print_ndarr(cob.asicPixelConfig, 'seg:%02d trbits: %s asicPixelConfig:'%(k, str(cob.trbit)))

          print('WORKS')
          for k,v in scfg.items():
              cob = v.config
              print_ndarr(cob.asicPixelConfig, 'seg:%02d trbits: %s asicPixelConfig:'%(k, str(cob.trbit)))
              

def issue_2020_12_19():
    """First version of det.raw.calib"""
    from psana.pyalgos.generic.NDArrUtils import print_ndarr, info_ndarr
    from psana import DataSource

    ds = DataSource(exp='ueddaq02',run=28)

    for run in ds.runs():
        print('====== run.runnum: ', run.runnum) 

        det = run.Detector('epixquad')

        print('det._det_name      : ', det._det_name) # epixquad
        print('det._dettype       : ', det._dettype)  # epix
        print('det.raw._det_name  : ', det.raw._det_name) # epixquad
        print('det.raw._dettype   : ', det.raw._dettype)  # epix
        print('det.raw._uniqueid  : ', det.raw._uniqueid)  # epix_3926196238-017....
        print('_sorted_segment_ids: ', det.raw._sorted_segment_ids) # [0, 1, 2, 3]

        for stepnum,step in enumerate(run.steps()):

          print('%s\n==== Step %1d ====' % (50*'_',stepnum))

          for evnum,evt in enumerate(step.events()):
            if evnum>2 and evnum%500!=0: continue
            print('%s\nStep %1d Event %04d' % (50*'_',stepnum, evnum))

            calib  = det.raw.calib(evt)
            ###########################
            print(info_ndarr(det.raw._pedestals(), 'peds  '))
            print(info_ndarr(det.raw.raw(evt),    'raw   '))
            print(info_ndarr(calib,               'calib '))


def issue_2021_01_11():
    """direct access to calibration constants for run=0
    """
    #d = wu.calib_constants_all_types(det, exp=None, run=None, time_sec=None, vers=None, url=cc.URL)
    from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types
    d = calib_constants_all_types("epixquad", exp="ueddaq02", run=0)
    print('\nd:',d)
    print(50*'-')
    print('d.keys():',d.keys())
    print(50*'-')


def issue_2021_02_03():
  """Data access example for confluence
     run, step, event loops
  """
  class Arguments:
    expt    = 'ueddaq02'
    run     = 66
    evtmax  = 5
    detname = 'epixquad'

  args = Arguments()

  from psana.pyalgos.generic.NDArrUtils import info_ndarr
  from psana import DataSource
  ds = DataSource(exp=args.expt, run=args.run, dir=f'/cds/data/psdm/{args.expt[:3]}/{args.expt}/xtc')

  for irun,run in enumerate(ds.runs()):
    print('\n==== %02d run: %d exp: %s detnames: %s' % (irun, run.runnum, run.expt, ','.join(run.detnames)))

    print('make %s detector object' % args.detname)
    det = run.Detector(args.detname)

    for istep,step in enumerate(run.steps()):
      print('\nStep %1d' % istep)

      for ievt,evt in enumerate(step.events()):
        if ievt>args.evtmax: exit('exit by number of events limit %d' % args.evtmax)

        print('%s\nEvent %04d' % (80*'_',ievt))
        segs = det.raw.segments(evt)
        raw  = det.raw.raw(evt)

        print(info_ndarr(segs, 'segsments '))
        print(info_ndarr(raw,  'raw '))


def issue_2021_02_08():
  """
     Issue: enumerate(run.steps()) DOES NOT WORK FOR RUN 65 and other
     problem arise due to misconfiguration of ueddaq02, runnum = 27 # WORKS, runnum = 65 # DOES NOT WORK
     solution: DataSource(files=...)
  """
  from psana import DataSource
  #runnum = 27 # WORKS
  runnum = 65 # DOES NOT WORK
  #ds = DataSource(exp='ueddaq02', run=runnum) #, dir=f'/cds/data/psdm/ued/ueddaq02/xtc', max_events=1000)
  ds = DataSource(files='/cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0065-s001-c000.xtc2') #86-charge injection
  for irun,run in enumerate(ds.runs()):
    print('\n==== %02d run: %d exp: %s detnames: %s' % (irun, run.runnum, run.expt, ','.join(run.detnames)))

    det = run.Detector('epixquad')
    step_docstring = run.Detector('step_docstring')
    print('step_docstring is created')

    for istep,step in enumerate(run.steps()):
      print('\nStep %1d' % istep)
      print('  step metadata: %s' % str(step_docstring(step)))


def issue_2021_02_09():
  """
     Issue: Peck, Ariana N <apeck@slac.stanford.edu> Thu 1/28/2021 2:47 PM
     What?s the equivalent function to the det.calib(evt) method in psana2?
  """
  from psana.pyalgos.generic.NDArrUtils import info_ndarr
  from psana import DataSource

  ds = DataSource(exp='tmoc00118', run=123, max_events=100)
  run = next(ds.runs())
  det = run.Detector('tmoopal')

  print('det._det_name      : ', det._det_name) # epixquad
  print('det._dettype       : ', det._dettype)  # epix
  print('det.raw._det_name  : ', det.raw._det_name) # epixquad
  print('det.raw._dettype   : ', det.raw._dettype)  # epix
  print('det.raw._uniqueid  : ', det.raw._uniqueid)  # epix_3926196238-017....
  print('_sorted_segment_ids: ', det.raw._sorted_segment_ids) # [0, 1, 2, 3]

  for evt in run.events():
    raw = det.raw.raw(evt)
    #calib = det.raw.calib(evt)
    print(info_ndarr(raw, 'raw ', last=10))
    #print(info_ndarr(calib, 'calib '))


#if __name__ == "__main__":
USAGE = '\nUsage:'\
      + '\n  python %s <test-name>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - issue_2020_11_09 - cpo something about epix10k2M/quad raw'\
      + '\n    2 - issue_2020_11_24 - cpo opal access to consatants for the same detector but other experiment'\
      + '\n    3 - issue_2020_12_02 - new file for epixquad issue with ds.runs loors like event loop ????'\
      + '\n    4 - issue_2020_12_10 - cpo access metadata from step'\
      + '\n    5 - issue_2020_12_16 - Chris access to config does not work, Matts works'\
      + '\n    6 - issue_2020_12_19 - First version of det.raw.calib'\
      + '\n    7 - issue_2021_01_11 - access to calibration constants for run=0 ... det.raw.calib'\
      + '\n    8 - issue_2021_02_03 - Data access example for confluence'\
      + '\n    9 - issue_2021_02_08 - docstring access is crashing'\
      + '\n   10 - issue_2021_02_09 - Peck, Ariana - missing calibration for opal detector'\

TNAME = sys.argv[1] if len(sys.argv)>1 else '0'

if   TNAME in  ('1',): issue_2020_11_09()
elif TNAME in  ('2',): issue_2020_11_24()
elif TNAME in  ('3',): issue_2020_12_02()
elif TNAME in  ('4',): issue_2020_12_10()
elif TNAME in  ('5',): issue_2020_12_16()
elif TNAME in  ('6',): issue_2020_12_19()
elif TNAME in  ('7',): issue_2021_01_11()
elif TNAME in  ('8',): issue_2021_02_03()
elif TNAME in  ('9',): issue_2021_02_08()
elif TNAME in ('10',): issue_2021_02_09()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

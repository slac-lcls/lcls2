def test_ipython():

    print('DATA FILE IS AVAILABLE ON drp-ued-cmp001 ONLY')

    #from psana2.pyalgos.generic.NDArrUtils import info_ndarr
    from psana import DataSource

    ds = DataSource(files='/u2/pcds/pds/ued/ueddaq02/xtc/ueddaq02-r0028-s000-c000.xtc2')
    run = next(ds.runs())

    det = run.Detector('epixquad')
    step = next(run.steps())

    evt = next(step.events())

    v = det.step.value(evt)
    d = det.step.docstring(evt)

    detsd = run.Detector('step_docstring') #Out[6]: <psana.detector.envstore.scan_raw_2_0_0 at 0x7f1a24735c10>
    detsv = run.Detector('step_value') #Out[8]: <psana.detector.envstore.scan_raw_2_0_0 at 0x7f1a0b205c10>

    from psana import DataSource
    ds = DataSource(exp='tmoc00118', run=123, max_events=100)
    run = next(ds.runs())
    det = run.Detector('tmoopal')
    print('run.dsparms.det_classes dict content:\n  %s' % str(run.dsparms.det_classes))



    run = None
    evt = None
    from psana import DataSource
    ds = DataSource(exp='ascdaq18', run=24, max_events=100)
    print('ds.xtc_files:\n ', '\n  '.join(ds.xtc_files))

    for irun,run in enumerate(ds.runs()):
      print('\n==== %02d run: %d exp: %s detnames: %s' % (irun, run.runnum, run.expt, ','.join(run.detnames)))
      det = run.Detector('epixhr')
      print('det.raw._fullname       :', det.raw._fullname())

      for istep,step in enumerate(run.steps()):
        print('\nStep %02d' % istep, type(step), end='')

        for ievt,evt in enumerate(step.events()):
          if ievt>10: continue #exit('exit by number of events limit %d' % args.evtmax)
          print('\n  Event %02d' % (ievt))

        st = evt.run().step(evt)
        print('XXX dir(st):', dir(st))

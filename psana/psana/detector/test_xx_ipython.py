def test_ipython():

    print('DATA FILE IS AVAILABLE ON drp-ued-cmp001 ONLY')

    #from psana.pyalgos.generic.NDArrUtils import info_ndarr
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

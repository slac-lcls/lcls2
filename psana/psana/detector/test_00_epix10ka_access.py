#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] %(message)s', level=logging.DEBUG)

from psana.pyalgos.generic.NDArrUtils import info_ndarr # print_ndarr
from psana import DataSource

import sys

SCRNAME = sys.argv[0].rsplit('/')[-1]
USAGE = '\n  python %s <test-name>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - quad'\
      + '\n    1 - quad segs 1 and 3'\
      + '\n    2 - opal - test tutorials/ami2/tmo/pop_raw_imgs.xtc2'\
      + '\n    3 - opal - real exp:tmolw0518, run:102, detname:tmoopal'\
      + '\n    4 - opal - real exp:tmolw0618, run:52, detname:tmoopal'\
      + '\n    5 - epixquasd - exp:tstx00117, run:144 - pedestals calibration run.config'\
      + '\n    6 - epixquasd - exp:ueddaq02,  run:27,28 - pedestals calibration det.config'\
      + '\n    7 - epixquasd - exp:ueddaq02,  run:27,28 - another version'\
      + '\n    8 - epixquasd - exp:ueddaq02,  run:27,28 - metadata from step'\

#print(USAGE)

tname = sys.argv[1] if len(sys.argv)>1 else '100'


def datasource_run(**kwa):
    ds = DataSource(**kwa)
    return ds, next(ds.runs())


def datasource_run_det(**kwa):
    ds = DataSource(**kwa)
    run = next(ds.runs())
    return ds, run, run.Detector(kwa.get('detname','opal'))


if tname in ('0','1'):

    fname = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005.xtc2' if tname ==0 else\
            '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005-seg1and3.xtc2'

    ds,run,det = datasource_run_det(files=fname, detname='epix10k2M')

    print('XXX dir(det):\n', dir(det))
    print('XXX dir(run):\n', dir(run))

    raw = det.raw
    print('dir(det.raw):', dir(det.raw))

    print('raw._configs:', raw._configs)
    cfg = raw._configs[0]
    print('dir(cfg):', dir(cfg))

    c0 = cfg.epix10k2M[0]
    print('dir(c0):', dir(c0))

    print('dir(c0.raw):', dir(c0.raw))

    print('WHAT IS THAT? c0.raw.raw', c0.raw.raw)


elif tname in('2','3','4'):

    kwa = {'files':'/reg/g/psdm/tutorials/ami2/tmo/pop_raw_imgs.xtc2'} if tname=='2' else\
          {'exp':'tmolw0518', 'run':102, 'detname':'tmoopal'} if tname=='3' else\
          {'exp':'tmolw0618', 'run':52, 'detname':'tmoopal'}

    ds,run,det = datasource_run_det(**kwa)

    print('XXX dir(det):\n', dir(det))
    print('XXX dir(run):\n', dir(run))

    print('XXX run.runnum  : ', run.runnum)   # 101
    print('XXX run.detnames: ', run.detnames) # {'opal'}
    print('XXX run.expt    : ', run.expt)     # amox27716
    print('XXX run.id      : ', run.id)       # 0

    print('XXX det.calibconst.keys(): ', det.calibconst.keys())   # dict_keys(['pop_rbfs'])
    #print('XXX det.calibconst[pop_rbfs]: ', det.calibconst['pop_rbfs']) #

    resp =  det.calibconst['pop_rbfs']
    print('XXX len(det.calibconst[pop_rbfs]): ', len(resp))
    rbfs_cons, rbfs_meta = resp

    for k,v in rbfs_cons.items():
        if k>20: break
        print(info_ndarr(v,'%03d: '%k, last=5))

    print('XXX type(rbfs_cons): ', type(rbfs_cons)) # <class 'dict'>
    print('XXX type(rbfs_meta): ', type(rbfs_meta)) # <class 'dict'>
    print('XXX rbfs_meta: ', rbfs_meta) #

elif tname == '5':

    #on daq-det-drp01:
    #detnames /u2/lcls2/tst/tstx00117/xtc/tstx00117-r0144-s000-c000.xtc2
    #Name     | Data Type
    #--------------------
    #epixquad | raw      

    # cdb add -e tstx00117 -d epixquad -c geometry -r0 -f /reg/g/psdm/detector/alignment/epix10kaquad/2020-11-20-epix10kaquad0-ued/2020-11-20-epix10kaquad0-ued-geometry.txt

    print('DATA FILE IS AVAILABLE ON daq-det-drp01 ONLY')
    ds,run,det = datasource_run_det(files='/u2/lcls2/tst/tstx00117/xtc/tstx00117-r0144-s000-c000.xtc2', detname='epixquad')

    print('\nXXX dir(run):\n', dir(run))

    print('XXX run.runnum  : ', run.runnum)   # 144
    print('XXX run.detnames: ', run.detnames) # {'epixquad'}
    print('XXX run.expt    : ', run.expt)     # tstx00117
    print('XXX run.id      : ', run.id)       # 0

    print('\nXXX dir(det):\n', dir(det))
    print('XXX det.calibconst.keys(): ', det.calibconst.keys())   # dict_keys(['geometry'])
    #print(det.calibconst)
    print('XXX det._det_name: ', det._det_name) # epixquad
    print('XXX det._dettype : ', det._dettype)  # epix
    print('XXX det._detid   : ', det._detid)    # -

    print('run.configs:', run.configs)          # [<dgram.Dgram object at 0x7fac52b5ddf0>] ???? WHY IT IS A LIST? HOW TO GET LIST INDEX FOR DETECTOR?
    cfg = run.configs[0]
    print('\ndir(cfg):', dir(cfg))              # [..., 'epixquad', 'service', 'software', 'timestamp']

    c0 = cfg.epixquad[0]                        # WHAT ????
    print('\ndir(c0):', dir(c0))                # [..., 'config']

    cfd = c0.config
    print('\ndir(c0.config):', dir(cfd))        # [..., 'asicPixelConfig', 'expert', 'trbit', 'user']
    print(info_ndarr(cfd.asicPixelConfig,'cfd.asicPixelConfig: ', last=10)) # PER ASIC ARRAY (4, 178, 192) !!!
                                                # shape:(4, 178, 192) size:136704 dtype:uint8 [12 12 12...
    print('cfd.trbit :', cfd.trbit)             # [1 1 1 1]
    print('cfd.user  :', cfd.user)              # <container.Container object at 0x7f284c06db70>
    print('cfd.expert:', cfd.expert)            # <container.Container object at 0x7f284c06db70>

    cfuser = cfd.user
    print('\ndir(cfd.user): ', dir(cfuser))     # [..., 'gain_mode', 'pixel_map', 'start_ns']
    print(info_ndarr(cfuser.pixel_map,'cfuser.pixel_map: ', last=10))
                                                # shape:(16, 178, 192) size:546816 dtype:uint8 [12 12 12...
    print('cfuser.start_ns:', cfuser.start_ns)  # 107749
    print('cfuser.gain_mode:', cfuser.gain_mode)# <container.Container object at 0x7fb005426b50>
    cfgm = cfuser.gain_mode
    print('\ndir(cfuser.gain_mode):', dir(cfgm)) # [..., 'names', 'value']
    print('cfgm.names:', cfgm.names)             # {3: 'AutoHiLo', 4: 'AutoMedLo', 0: 'High', 2: 'Low', 5: 'Map', 1: 'Medium'} WHAT IS MAP???
    print('cfgm.value:', cfgm.value)             # 0

    cfexpert = cfd.expert
    print('\ndir(cfd.cfexpert): ', dir(cfexpert))# 


elif tname == '6':

    #print('DATA FILE IS AVAILABLE ON daq-det-drp01 ONLY')
    #ds,run,det = datasource_run_det(files='/u2/lcls2/tst/tstx00117/xtc/tstx00117-r0147-s000-c000.xtc2', detname='epixquad')

    print('DATA FILE IS AVAILABLE ON drp-ued-cmp001 ONLY')
    ds,run,det = datasource_run_det(files='/u2/pcds/pds/ued/ueddaq02/xtc/ueddaq02-r0028-s000-c000.xtc2', detname='epixquad')

    print('\nXXX dir(run):\n', dir(run)) # ['Detector', ..., '_check_empty_calibconst', '_evt', '_evt_iter', '_get_runinfo', 'analyze', 'beginruns', 'c_ana', 'configs', 'detinfo', 'detnames', 'dm', 'dsparms', 'epicsinfo', 'esm', 'events', 'expt', 'id', 'nfiles', 'run', 'runnum', 'scan', 'scaninfo', 'smd_dm', 'smd_fds', 'stepinfo', 'steps', 'timestamp', 'xtcinfo']

    print('XXX run.runnum  : ', run.runnum)   # 144
    print('XXX run.detnames: ', run.detnames) # {'epixquad'}
    print('XXX run.expt    : ', run.expt)     # tstx00117
    print('XXX run.id      : ', run.id)       # 0
    print('XXX run.timestamp: ', run.timestamp) # 4190613356186573936
    print('XXX run.steps   : ', run.steps())    # <generator object RunSingleFile.steps at 0x7f3e1932b1d0>
    print('XXX run.events(): ', run.events())   # <generator object RunSingleFile.events at 0x7f3e1932b1d0>

    evt = None
    for evt in run.events():
        if evt is not None: break


    print('\nXXX dir(det):\n', dir(det))      # [..., '_configs', '_det_name', '_detid', '_dettype', 'calibconst', 'raw', 'step']
    print('XXX det.calibconst.keys(): ', det.calibconst.keys())   # dict_keys(['geometry'])
    #print(det.calibconst)
    print('XXX det._det_name: ', det._det_name) # epixquad
    print('XXX det._dettype : ', det._dettype)  # epix
    print('XXX det._detid   : ', det._detid)    # -

    print('\nXXX dir(det.raw):', dir(det.raw)) # [..., '_add_fields', '_calibconst', '_configs', '_det_name', '_dettype', '_drp_class_name', '_env_store', '_info', '_return_types', '_seg_configs', '_segments', '_sorted_segment_ids', '_uniqueid', '_var_name', 'array']

    print('XXX det.raw._det_name: ', det.raw._det_name) # epixquad
    print('XXX det.raw._dettype : ', det.raw._dettype)  # epix
    print('XXX det.raw._calibconst.keys(): ', det.raw._calibconst.keys()) # dict_keys(['geometry'])
    print('XXX det.raw._segments(evt): ', det.raw._segments(evt)) # {0: <container.Container object at 0x7f29b7db4b30>, 1: <container.Container object at 0x7f29b7db4b90>, 2: <container.Container object at 0x7f29b7db4bf0>, 3: <container.Container object at 0x7f29b7db4c50>}
    print('XXX det.raw._seg_configs(): ', det.raw._seg_configs()) # {0: <container.Container object at 0x7f29b7df2870>, 1: <container.Container object at 0x7f29b7df28d0>, 2: <container.Container object at 0x7f29b7df2930>, 3: <container.Container object at 0x7f29b7df2990>}
 
    print('XXX det.raw._uniqueid: ', det.raw._uniqueid)
                                                # epix_3926196238-0175152897-1157627926-0000000000-0000000000-0000000000-0000000000_-_3926196238-0175552257-3456106518-0000000000-0000000000-0000000000-0000000000_3926196238-0176373505-4043309078-0000000000-0000000000-0000000000-0000000000
    print('XXX det.raw._sorted_segment_ids: ', det.raw._sorted_segment_ids) # [0, 1, 2, 3]

    seg0 = det.raw._segments(evt)[0]
    cfg0 = det.raw._seg_configs()[0]

    print('\nXXX dir(seg0):', dir(seg0))      # [..., 'aux', 'raw']
    print(info_ndarr(seg0.raw,'XXX seg0.raw: ', last=10)) #shape:(352, 384) size:135168 dtype:uint16 [2980 2942 3021 3047 ...
    print(info_ndarr(seg0.aux,'XXX seg0.aux: ', last=10)) #shape:(4, 384) size:1536 dtype:uint16 [2699 2867 2619 2853 2651 ...

    print('\nXXX dir(cfg0):', dir(cfg0))      # [..., '_xtc', 'config']
    print('XXX cfg0.config:', cfg0.config)    # <container.Container object at 0x7fefe157a8d0>
    print('\nXXX dir(cfg0.config):', dir(cfg0.config))  #[..., 'asicPixelConfig', 'trbit']
    print(info_ndarr(cfg0.config.asicPixelConfig,'cfg0.config.asicPixelConfig: ', last=10)) # shape:(4, 178, 192) size:136704 dtype:uint8 [12 12 12
    print(info_ndarr(cfg0.config.trbit,'cfg0.config.trbit: ', last=10)) # shape:(4,) size:4 dtype:uint8 [1 1 1 1]

    cfgs = det._configs

    print('det._configs:', cfgs)                # [<dgram.Dgram object at 0x7f5a36a1bd40>]
    cfg = cfgs[0]
    print('\ndir(cfg):', dir(cfg))              # [..., 'epixquad', 'service', 'software', 'timestamp'] 

    cfquad = cfg.epixquad                       # WHAT, epixquad ????
    print('c0=cfg.epixquad :', cfquad)          # {0: <container.Container object at 0x7f79cfdd5fd0>, 
                                                #  1: <container.Container object at 0x7f79cfd9cbb0>, 
                                                #  2: <container.Container object at 0x7f79cfd9cc10>,
                                                #  3: <container.Container object at 0x7f79cfd9cc70>} # PER PANEL?
    cfp0 = cfquad[0]
    print('\ndir(cfp0):', dir(cfp0))            # [..., '_xtc', 'config']

    cfp = cfp0.config
    print('\ndir(cfp):', dir(cfp))              # [..., 'asicPixelConfig', 'expert', 'trbit', 'user']

    print(info_ndarr(cfp.asicPixelConfig,'cfp.asicPixelConfig: ', last=10)) # PER ASIC ARRAY (4, 178, 192) !!! 
                                                # shape:(4, 178, 192) size:136704 dtype:uint8 [12 12 ...
    print('cfp.trbit :', cfp.trbit)             # [1 1 1 1]


    if False:
        print('cfp.user  :', cfp.user)              # <container.Container object at 0x7f284c06db70>
        print('cfp.expert:', cfp.expert)            # <container.Container object at 0x7f284c06db70>
        
        cfuser = cfp.user
        print('\ndir(cfuser):', dir(cfuser))        # [..., 'gain_mode', 'pixel_map', 'start_ns'
        print(info_ndarr(cfuser.pixel_map,'cfuser.pixel_map: ', last=10))
                                                    # shape:(16, 178, 192) size:546816 dtype:uint8 [12 12 12...
        print('cfuser.start_ns:', cfuser.start_ns)  # 107749
        print('cfuser.gain_mode:', cfuser.gain_mode)# <container.Container object at 0x7fb005426b50>
        cfgm = cfuser.gain_mode
        print('\ndir(cfuser.gain_mode):', dir(cfgm)) # [..., 'names', 'value']
        print('cfgm.names:', cfgm.names)             # {3: 'AutoHiLo', 4: 'AutoMedLo', 0: 'High', 2: 'Low', 5: 'Map', 1: 'Medium'} WHAT IS MAP???
        print('cfgm.value:', cfgm.value)             # 0
        
        cfexpert = cfp.expert
        print('\ndir(cfexpert):', dir(cfexpert))      # ['DevPcie', 'EpixQuad', ...]
        print('cfexpert.DevPcie:', cfexpert.DevPcie)  # <container.Container object at 0x7f4ba0e05070>
        print('cfexpert.EpixQuad:', cfexpert.EpixQuad)# <container.Container object at 0x7feee97901b0>
        
        print('\ndir(cfexpert.DevPcie):', dir(cfexpert.DevPcie))  # ['Hsio', ...]
        print('\ndir(cfexpert.EpixQuad):', dir(cfexpert.EpixQuad))# ['AcqCore', 'Ad9249Config[7]', 'Ad9249Readout[0]', 'Ad9249Readout[1]', 'Ad9249Readout[2]', 'Ad9249Readout[3]', 'Ad9249Readout[4]', 'Ad9249Readout[5]', 'Ad9249Readout[6]', 'Ad9249Readout[7]', 'Ad9249Readout[8]', 'Ad9249Readout[9]', 'Ad9249Tester', 'Epix10kaSaci[0]', 'Epix10kaSaci[10]', 'Epix10kaSaci[11]', 'Epix10kaSaci[12]', 'Epix10kaSaci[13]', 'Epix10kaSaci[14]', 'Epix10kaSaci[15]', 'Epix10kaSaci[1]', 'Epix10kaSaci[2]', 'Epix10kaSaci[3]', 'Epix10kaSaci[4]', 'Epix10kaSaci[5]', 'Epix10kaSaci[6]', 'Epix10kaSaci[7]', 'Epix10kaSaci[8]', 'Epix10kaSaci[9]', 'PseudoScopeCore', 'RdoutCore', 'SystemRegs', 'VguardDac', ...]


elif tname == '7':
    #print('DATA FILE IS AVAILABLE ON daq-det-drp01 ONLY')
    #fname = '/u2/lcls2/tst/tstx00117/xtc/tstx00117-r0147-s000-c000.xtc2'

    print('DATA FILE IS AVAILABLE ON drp-ued-cmp001 ONLY')
    fname = '/u2/pcds/pds/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2'
    detname='epixquad'

    ds = DataSource(files=fname)
    print('\nXXX dir(ds):', dir(ds)) #[...,'_abc_impl', '_close_opened_smd_files', '_configs', '_end_prometheus_client', '_get_runinfo', '_set_configinfo', '_setup_beginruns', '_setup_det_class_table', '_setup_run', '_setup_run_calibconst', '_setup_run_files', '_setup_runnum_list', '_start_prometheus_client', '_start_run', 'batch_size', 'destination', 'detectors', 'dir', 'dm', 'dsparms', 'exp', 'files', 'filter', 'live', 'max_events', 'monitor', 'prom_man', 'runnum', 'runnum_list', 'runnum_list_index', 'runs', 'shmem', 'smalldata', 'smalldata_kwargs']
    print('XXX ds.runnum: ', ds.runnum) # None
    print('XXX ds.exp   : ', ds.exp)    # None
    print('XXX ds.runs(): ', ds.runs()) # 

    #for run in ds.runs(): # DOES NOT WORK
    run = next(ds.runs())
    print('\nXXX dir(run):', dir(run)) # ['Detector', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_check_empty_calibconst', '_evt', '_evt_iter', '_get_runinfo', 'analyze', 'beginruns', 'c_ana', 'configs', 'detinfo', 'detnames', 'dm', 'dsparms', 'epicsinfo', 'esm', 'events', 'expt', 'id', 'nfiles', 'run', 'runnum', 'scan', 'scaninfo', 'smd_dm', 'smd_fds', 'stepinfo', 'steps', 'timestamp', 'xtcinfo']
    print('XXX runnum       : ', run.runnum)   # 147
    print('XXX run.detnames : ', run.detnames) # {'epixquad'}
    print('XXX run.expt     : ', run.expt)     # tstx00117
    print('XXX run.id       : ', run.id)       # 0
    print('XXX run.timestamp: ', run.timestamp)# 4190613356186573936
    #print('XXX run.run()   : ', run.run())    # DOES NOT WORK
    print('XXX run.stepinfo : ', run.stepinfo) # {('epixquad', 'step'): ['value', 'docstring'], ('epixquadhw', 'step'): ['value', 'docstring']}
    print('XXX run.steps()  : ', run.steps())  # <generator object RunSingleFile.steps at 0x7fdc4bf8ead0>

    det = run.Detector(detname)
    print('\nXXX dir(det):', dir(det))          #[...,'_configs', '_det_name', '_detid', '_dettype', 'calibconst', 'raw', 'step']
    print('\nXXX dir(det.raw):', dir(det.raw))  #[...,'_add_fields', '_calibconst', '_configs', '_det_name', '_dettype', '_drp_class_name', '_env_store', '_info', '_return_types', '_seg_configs', '_segments', '_sorted_segment_ids', '_uniqueid', '_var_name', 'array']
    print('\nXXX dir(det.step):', dir(det.step))#[...,'dgrams', 'docstring', 'env_store', 'value']

    print('XXX det.calibconst.keys(): ', det.calibconst.keys()) # dict_keys(['geometry'])
    #print(det.calibconst)
    print('XXX det._det_name: ', det._det_name) # epixquad
    print('XXX det._dettype : ', det._dettype)  # epix
    print('XXX det._detid   : ', det._detid)    # -

    print('run.configs:', run.configs)          # [<dgram.Dgram object at 0x7fac52b5ddf0>] ???? WHY IT IS A LIST? HOW TO GET LIST INDEX FOR DETECTOR?
    cfg = run.configs[0]
    print('\ndir(cfg):', dir(cfg))              # [..., 'epixquad', 'service', 'software', 'timestamp']

    for stepnum,step in enumerate(run.steps()):
        print('\nSTEP dir(step):', dir(step))   # [..., 'esm', 'events', 'evt', 'evt_iter']    
        print('STEP step.esm  : ', step.esm)    # <psana.psexp.envstore_manager.EnvStoreManager object at 0x7fd3ab1fffd0>
        print('\nSTEP dir(step.esm):', dir(step.esm))   # [..., '_update_config', 'configs', 'env_from_variable', 'get_stepinfo', 'stores', 'update_by_event', 'update_by_views']
        print('\nSTEP dir(step.evt):', dir(step.evt))   # [..., '__weakref__', '_assign_det_segments', '_complete', '_det_segments', '_dgrams', '_from_bytes', '_has_offset', '_nanoseconds', '_position', '_replace', '_run', '_seconds', '_size', '_to_bytes', 'datetime', 'get_offsets_and_sizes', 'next', 'run', 'service', 'timestamp']
        print('STEP step.esm.get_stepinfo():', step.esm.get_stepinfo()) #  {('epixquadhw', 'step'): ['value', 'docstring'], ('epixquad', 'step'): ['value', 'docstring']}
        print('step.esm.configs', step.esm.configs) # [<dgram.Dgram object at 0x7f207c56bc90>]
        print('\nSTEP dir(step.esm.configs[0]):',  dir(step.esm.configs[0]))  # [..., '_dgrambytes', '_file_descriptor', '_offset', '_size', '_xtc', 'epixquad', 'epixquadhw', 'service', 'software', 'timestamp']

        cfgs0 = step.esm.configs[0]
        print('STEP dir(step.esm.configs[0].epixquad[0]):', dir(cfgs0.epixquad[0])) # {0: <container.Container object at 0x7fb10ae68d10>, 1: ... = [..., 'config']
        print('STEP step.esm.configs[0]._file_descriptor:', cfgs0._file_descriptor) # 19
        print('STEP step.esm.configs[0]._offset:', cfgs0._offset) # 0
        print('STEP step.esm.configs[0]._size:', cfgs0._size) # 1819055
        #print('STEP step.esm.configs[0]._dgrambytes:', cfgs0._dgrambytes) #
        print('STEP dir(step.esm.configs[0].epixquadhw[0]):', dir(cfgs0.epixquadhw[0])) # {0: <container.Container object at 0x7fb10af7f170>} = [..., 'config']

        print('STEP step.esm.configs[0].service():', cfgs0.service()) # 2
        print('STEP dir(step.esm.configs[0].software):', dir(cfgs0.software)) # <container.Container object at 0x7fb10b1da1d0> = [..., 'epixquad', 'epixquadhw', 'runinfo']
        print('STEP step.esm.configs[0].timestamp():', cfgs0.timestamp()) # 4190613348438778143


        #exit('TEST EXIT')
        #################

        for evnum,evt in enumerate(step.events()):
            if evnum>2 and evnum%500!=0: continue
            print('%s\nStep %1d Event %04d' % (50*'_',stepnum, evnum))
            segs = det.raw._segments(evt)
            raw  = det.raw.raw(evt)
            logger.info('segs: %s' % str(segs))
            logger.info(info_ndarr(raw,  'raw  '))
        print(50*'-')


elif tname == '8':
    print('DATA FILE IS AVAILABLE ON drp-ued-cmp001 ONLY')
    from psana import DataSource
    import sys
    #ds = DataSource(exp='ueddaq02',run=28)
    fname = '/u2/pcds/pds/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2'
    detname='epixquad'
    ds = DataSource(files=fname)
    myrun = next(ds.runs())
    step_value = myrun.Detector('step_value')
    step_docstring = myrun.Detector('step_docstring')
    for nstep,step in enumerate(myrun.steps()):
        print('step:',nstep,step_value(step),step_docstring(step))
        for nevt,evt in enumerate(step.events()):
            if nevt==3: print('evt3:',nstep,step_value(evt),step_docstring(evt))


else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%tname)

exit('END OF TEST %s'%tname)

#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] %(message)s', level=logging.DEBUG)

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
      + '\n    6 - epixquasd - exp:tstx00117, run:147 - pedestals calibration det.config'\

print(USAGE)

tname = sys.argv[1] if len(sys.argv)>1 else '2'


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

    from psana.pyalgos.generic.NDArrUtils import info_ndarr # print_ndarr

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
    from psana.pyalgos.generic.NDArrUtils import info_ndarr # print_ndarr

    #on daq-det-drp01:
    #detnames /u2/lcls2/tst/tstx00117/xtc/tstx00117-r0144-s000-c000.xtc2
    #Name     | Data Type
    #--------------------
    #epixquad | raw      

    # cdb add -e tstx00117 -d epixquad -c geometry -r0 -f /reg/g/psdm/detector/alignment/epix10kaquad/2020-11-20-epix10kaquad0-ued/2020-11-20-epix10kaquad0-ued-geometry.txt

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
    from psana.pyalgos.generic.NDArrUtils import info_ndarr # print_ndarr

    ds,run,det = datasource_run_det(files='/u2/lcls2/tst/tstx00117/xtc/tstx00117-r0147-s000-c000.xtc2', detname='epixquad')

    print('\nXXX dir(run):\n', dir(run))

    print('XXX run.runnum  : ', run.runnum)   # 144
    print('XXX run.detnames: ', run.detnames) # {'epixquad'}
    print('XXX run.expt    : ', run.expt)     # tstx00117
    print('XXX run.id      : ', run.id)       # 0

    print('\nXXX dir(det):\n', dir(det))      # [..., '_configs', '_det_name', '_detid', '_dettype', 'calibconst', 'raw', 'step']
    print('XXX det.calibconst.keys(): ', det.calibconst.keys())   # dict_keys(['geometry'])
    #print(det.calibconst)
    print('XXX det._det_name: ', det._det_name) # epixquad
    print('XXX det._dettype : ', det._dettype)  # epix
    print('XXX det._detid   : ', det._detid)    # -

    raw = det.raw
    print('\nXXX dir(det.raw):\n', dir(raw)) # [..., '_add_fields', '_calibconst', '_configs', '_det_name', '_dettype', '_drp_class_name', '_env_store', '_info', '_return_types', '_segments', '_sorted_segment_ids', '_uniqueid', '_var_name', 'array']

    
    print('XXX raw._uniqueid: ', raw._uniqueid)
                                                # epix_3926196238-0175152897-1157627926-0000000000-0000000000-0000000000-0000000000_-_3926196238-0175552257-3456106518-0000000000-0000000000-0000000000-0000000000_3926196238-0176373505-4043309078-0000000000-0000000000-0000000000-0000000000
    print('XXX raw._sorted_segment_ids: ', raw._sorted_segment_ids) # [0, 1, 2, 3]

    print('XXX raw._det_name: ', raw._det_name) # epixquad
    print('XXX raw._dettype : ', raw._dettype)  # epix
 
    #exit()
    ######


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

else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%tname)

exit('END OF TEST %s'%tname)

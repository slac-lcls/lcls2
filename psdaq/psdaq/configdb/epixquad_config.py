from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.cas.xpm_utils import timTxId
from .xpmmini import *
import rogue
import epix_l2sidaq
import ePixQuad
import lcls2_pgp_pcie_apps
import time
import json
import os
import numpy as np
import IPython
from collections import deque

base = None
pv = None
lane = 0
chan = None
group = None
ocfg = None

def mode(a):
    uniqueValues = np.unique(a).tolist()
    uniqueCounts = [len(np.nonzero(a == uv)[0])
                    for uv in uniqueValues]

    modeIdx = uniqueCounts.index(max(uniqueCounts))
    return uniqueValues[modeIdx]

def dumpvars(prefix,c):
    print(prefix)
    for key,val in c.nodes.items():
        name = prefix+'.'+key
        dumpvars(name,val)

#
#  Apply the configuration dictionary to the rogue registers
#
def apply_dict(pathbase,base,cfg):
    depth = 0
    my_queue  =  deque([[pathbase,depth,base,cfg]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        if(dict is type(configdb_node)):
            for i in configdb_node:
                try:
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[i],configdb_node[i]])
                except KeyError:
                    print('Lookup failed for node [{:}] in path [{:}]'.format(i,path))

        #  Apply
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path is not pathbase ):
#            if False:
            if (('Saci' in path and 'PixelDummy' in path) or
                ('Saci[3]' in path and 'CompEn' in path) or
                ('Saci[3]' in path and 'Preamp' in path) or
                ('Saci[3]' in path and 'MonostPulser' in path) or
                ('Saci[3]' in path and 'PulserDac' in path)):
                print(f'NOT setting {path} to {configdb_node}')
            else:
                print(f'Setting {path} to {configdb_node}')
                rogue_node.set(configdb_node)
#                time.sleep(0.0001)

#
#  Construct an asic pixel mask with square spacing
#
def pixel_mask_square(value0,value1,spacing,position):
    ny,nx=352,384;
    if position>=spacing**2:
        print('position out of range')
        position=0;
    out=np.zeros((ny,nx),dtype=np.int)+value0
    position_x=position%spacing; position_y=position//spacing
    out[position_y::spacing,position_x::spacing]=value1
    return out

#
#  Initialize the rogue accessor
#
def epixquad_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None):
    global base
    global pv
    global lane
    print('epixquad_init')

    base = {}
#    base['log'] = open('config.log','w')

    #  Configure the PCIe card first (timing, datavctap)
    if True:
        pbase = lcls2_pgp_pcie_apps.DevRoot(dev           =dev,
                                            enLclsI       =False,
                                            enLclsII      =True,
                                            startupMode   =True,
                                            standAloneMode=xpmpv is not None,
                                            pgp3          =True,
                                            dataVc        =0,
                                            pollEn        =False,
                                            initRead      =False,
                                            numLanes      =4,
                                            devTarget     =lcls2_pgp_pcie_apps.Kcu1500)
        #dumpvars('pbase',pbase)

        pbase.__enter__()

        # Open a new thread here
        if xpmpv is not None:
            #pbase.DevPcie.Hsio.TimingRx.ConfigureXpmMini()
            pv = PVCtrls(xpmpv,pbase.DevPcie.Hsio.TimingRx.XpmMiniWrapper)
            pv.start()
        else:
            #pbase.DevPcie.Hsio.TimingRx.ConfigLclsTimingV2()
            time.sleep(0.1)
        base['pci'] = pbase

    #  Connect to the camera
    cbase = ePixQuad.Top(dev=dev,hwType='datadev',lane=lane,pollEn=False)
    #dumpvars('cbase',cbase)
    cbase.__enter__()
    base['cam'] = cbase
    return base

#
#  Set the PGP lane
#
def epixquad_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

#
#  Set the local timing ID and fetch the remote timing ID
#
def epixquad_connect(base):

    if 'pci' in base:
        pbase = base['pci']
        rxId = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        print('RxId {:x}'.format(rxId))
        txId = timTxId('epixquad')
        print('TxId {:x}'.format(txId))
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    else:
        rxId = 0xffffffff

    # fetch the serial number
    # SystemRegs->CarrierIdLow/High[0:3]
    # {"DigitalCardId0"},
    # {"DigitalCardId1"},
    # {"AnalogCardId0"},
    # {"AnalogCardId1"},
    # {"CarrierId0"},
    # {"CarrierId1"}
    # 0x30-0x33,0x3b-0x3c

    epixquadid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixquadid

    return d

#
#  Translate the 'user' components of the cfg dictionary into 'expert' settings
#  The cfg dictionary may be partial (scanning), so the ocfg dictionary is
#  reference for the full set.
#
def user_to_expert(base, cfg, full=False):
    global ocfg
    global group
    global lane

    pbase = base['pci']

    d = {}
    hasUser = 'user' in cfg
    if (hasUser and 'start_ns' in cfg['user']):
        partitionDelay = getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
        rawStart       = cfg['user']['start_ns']
        triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        if triggerDelay < 0:
            print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
            raise ValueError('triggerDelay computes to < 0')

        d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[{lane}].TriggerDelay']=triggerDelay

    if (hasUser and 'gate_ns' in cfg['user']):
        triggerWidth = int(cfg['user']['gate_ns']/10)
        if triggerWidth < 1:
            print(f'triggerWidth {triggerWidth} ({cfg['user']['gate_ns']} ns)')
            raise ValueError('triggerWidth computes to < 1')

        d[f'expert.EpixQuad.AcqCore.AsicAcqWidth']=triggerWidth

    if full:
        d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[{lane}].Partition']=group

    pixel_map_changed = False
    a = None
    if (hasUser and ('gain_mode' in cfg['user'] or 
                     'pixel_map' in cfg['user'])):
        gain_mode = cfg['user']['gain_mode']
        if gain_mode==5:
            a  = cfg['user']['pixel_map']
        else:
            mapv  = (0xc,0xc,0x8,0x0,0x0)[gain_mode] # H/M/L/AHL/AML
            trbit = (0x1,0x0,0x0,0x1,0x0)[gain_mode]
            a  = (np.array(ocfg['user']['pixel_map'],dtype=np.uint8) & 0x3) | mapv
            a = a.reshape(-1).tolist()
            for i in range(16):
                d[f'expert.EpixQuad.Epix10kaSaci[{i}].trbit'] = trbit
        print('pixel_map len {}'.format(len(a)))
        d['user.pixel_map'] = a
        pixel_map_changed = True

    update_config_entry(cfg,ocfg,d)

    return pixel_map_changed

#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writePixelMap=True):
    # Clear the pipeline
    #getattr(pbase.DevPcie.Application,'AppLane[%d]'%lane).EventBuilder.Blowoff.set(True)

    # overwrite the low-level configuration parameters with calculations from the user configuration
    pbase = base['pci']
    if ('expert' in cfg and 'DevPcie' in cfg['expert']):
        apply_dict('pbase.DevPcie',pbase.DevPcie,cfg['expert']['DevPcie'])

    cbase = base['cam']

    #  Make list of enabled ASICs
    asics = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    #  Important that Asic IsEn is True while configuring and false when running
    for i in asics:
        print(f'Enabling ASIC {i}')
        saci = cbase.Epix10kaSaci[i]
        saci.enable.set(True)  # Saci disabled by default!
        saci.IsEn.set(True)

    if ('expert' in cfg and 'EpixQuad' in cfg['expert']):
        epixQuad = cfg['expert']['EpixQuad'].copy()
        #  Add write protection word to upper range
        if 'AcqCore' in epixQuad and 'AsicRoClkHalfT' in epixQuad['AcqCore']:
            epixQuad['AcqCore']['AsicRoClkHalfT'] |= 0xaaaa0000
        if 'RdoutCore' in epixQuad and 'AdcPipelineDelay' in epixQuad['RdoutCore']:
            epixQuad['RdoutCore']['AdcPipelineDelay'] |= 0xaaaa0000
        apply_dict('cbase',cbase,epixQuad)

    if writePixelMap:
        if 'user' in cfg and 'pixel_map' in cfg['user']:
            #  Write the pixel gain maps
            #  Would like to send a 3d array
            a = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
            pixelConfigMap = np.reshape(a,(16,178,192))
            if True:
                #
                #  Accelerated matrix configuration (~2 seconds)
                #
                #  Found that gain_mode is mapping to [M/M/L/M/M]
                #    Like trbit is always zero (Saci was disabled!)
                #
                core = cbase.SaciConfigCore
                core.enable.set(True)
                core.SetAsicsMatrix(json.dumps(pixelConfigMap.tolist()))
                core.enable.set(False)
            else:
                #
                #  Pixel by pixel matrix configuration (up to 15 minutes)
                #
                #  Found that gain_mode is mapping to [H/M/M/H/M]
                #    Like pixelmap is always 0xc
                #
                for i in asics:
                    saci = cbase.Epix10kaSaci[i]
                    saci.PrepareMultiConfig()
                    
                #  Set the whole ASIC to its most common value
                masic = {}
                for i in asics:
                    masic[i] = mode(pixelConfigMap[i])
                    saci = cbase.Epix10kaSaci[i]
                    saci.WriteMatrixData(masic)  # 0x4000 v 0x84000

                #  Now fix any pixels not at the common value
                banks = ((0xe<<7),(0xd<<7),(0xb<<7),(0x7<<7))
                for i in asics:
                    saci = cbase.Epix10kaSaci[i]
                    nrows = pixelConfigMap.shape[1]
                    ncols = pixelConfigMap.shape[2]
                    for y in range(nrows):
                        for x in range(ncols):
                            if pixeConfigMap[i,y,x]!=masic[i]:
                                if row >= (nrows>>1):
                                    mrow = row - (nrows>>1)
                                    if col < (ncols>>1):
                                        offset = 3
                                        mcol = col
                                    else:
                                        offset = 0
                                        mcol = col - (ncols>>1)
                                else:
                                    mrow = (nrows>>1)-1 - row
                                    if col < (ncols>>1):
                                        offset = 2
                                        mcol = (ncols>>1)-1 - col
                                    else:
                                        offset = 1
                                        mcol = (ncols-1) - col
                                bank = (mcol % (48<<2)) / 48
                                bankOffset = banks[bank]
                                saci.RowCounter(y)
                                saci.ColCounter(bankOffset | (mcol%48))
                                saci.WritePixelData(pixelConfigMap[i,y,x])

            print('SetAsicsMatrix complete')
        else:
            print('writePixelMap but no new map')
            print(cfg)

    #  Important that Asic IsEn is True while configuring and false when running
    for i in asics:
        saci = cbase.Epix10kaSaci[i]
        saci.IsEn.set(False)
        saci.enable.set(False)

#
#  Called on Configure
#
def epixquad_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global ocfg
    global group
    group = rog

    _checkADCs()
#    _resetSequenceCount()

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    #  Translate user settings to the expert fields
    user_to_expert(base, cfg, full=True)

    #  Apply the expert settings to the device
    config_expert(base, cfg)

    pbase = base['pci']
    #pbase.DevPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.HwEnable.set(True)
    #getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager,f'TriggerEventBuffer[{lane}]').MasterEnable.set(True)
    #getattr(pbase.DevPcie.Application,'AppLane[%d]'%lane).EventBuilder.Blowoff.set(False)
    pbase.StartRun()

    #  Capture the firmware version to persist in the xtc
    cbase = base['cam']
    firmwareVersion = cbase.AxiVersion.FpgaVersion.get()
    #cfg['firmwareVersion:RO'] = cbase.AxiVersion.FpgaVersion.get()
    #cfg['firmwareBuild:RO'  ] = cbase.AxiVersion.BuildStamp.get()

#    print('--Configuring AsicMatrix for injection--')
#    cbase.SetAsicMatrixTest()

    ocfg = cfg
    #return [json.dumps(cfg)]

    #
    #  Create the segment configurations from parameters required for analysis
    #
    trbit = [ cfg['expert']['EpixQuad'][f'Epix10kaSaci[{i}]']['trbit'] for i in range(16)]

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]


    a = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
    pixelConfigMap = np.reshape(a,(16,178,192))

    for seg in range(4):
        #  Construct the ID
        carrierId = [ cbase.SystemRegs.CarrierIdLow [seg].get(),
                      cbase.SystemRegs.CarrierIdHigh[seg].get() ]
        digitalId = [ 0, 0 ]
        analogId  = [ 0, 0 ]
        id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                          carrierId[0], carrierId[1],
                                                          digitalId[0], digitalId[1],
                                                          analogId [0], analogId [1])
        top = cdict()
        top.setAlg('config', [2,0,0])
        top.setInfo(detType='epix', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
        top.set('asicPixelConfig', pixelConfigMap[4*seg:4*seg+4,:176].tolist(), 'UINT8')  # only the rows which have readable pixels
        top.set('trbit'          , trbit[4*seg:4*seg+4], 'UINT8')
        scfg[seg+1] = top.typed_json()

    result = []
    for i in range(5):
        print('json seg {}  detname {}'.format(i, scfg[i]['detName:RO']))
        result.append( json.dumps(scfg[i]) )
    return result

def epixquad_unconfig(base):
    pbase = base['pci']
    #getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager,f'TriggerEventBuffer[{lane}]').MasterEnable.set(False)
    pbase.StopRun()
    return base

#
#  Build the set of all configuration parameters that will change 
#  in response to the scan parameters
#
def epixquad_scan_keys(update):
    print('epixquad_scan_keys')
    global ocfg
    global base
    cfg = {}
    copy_reconfig_keys(cfg,ocfg,json.loads(update))
    # Apply to expert
    pixelMapChanged = user_to_expert(base,cfg,full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]

    if pixelMapChanged:
        a = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
        pixelConfigMap = np.reshape(a,(16,178,192))
        trbit = [ cfg['expert']['EpixQuad'][f'Epix10kaSaci[{i}]']['trbit'] for i in range(16)]

        cbase = base['cam']
        firmwareVersion = cbase.AxiVersion.FpgaVersion.get()
        for seg in range(4):
            carrierId = [ cbase.SystemRegs.CarrierIdLow [seg].get(),
                          cbase.SystemRegs.CarrierIdHigh[seg].get() ]
            digitalId = [ 0, 0 ]
            analogId  = [ 0, 0 ]
            id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                              carrierId[0], carrierId[1],
                                                              digitalId[0], digitalId[1],
                                                              analogId [0], analogId [1])
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epix', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigMap[4*seg:4*seg+4].tolist(), 'UINT8')
            top.set('trbit'          , trbit[4*seg:4*seg+4], 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

#    for i in range(len(result)):
#        base['log'].write('--scan keys-- {}\n'.format(i))
#        base['log'].write(result[i])

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixquad_update(update):
    print('epixquad_update')
    global ocfg
    global base
    # extract updates
    cfg = {}
    update_config_entry(cfg,ocfg,json.loads(update))
    #  Apply to expert
    writePixelMap = user_to_expert(base,cfg,full=False)
    #  Apply config
    config_expert(base, cfg, writePixelMap)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)

    topname = cfg['detName:RO'].split('_')

    scfg = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]

    scfg[0] = cfg

    if writePixelMap:
        a = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
        pixelConfigMap = np.reshape(a,(16,178,192))
        try:
            trbit = [ cfg['expert']['EpixQuad'][f'Epix10kaSaci[{i}]']['trbit'] for i in range(16)]
        except:
            trbit = None

        cbase = base['cam']
        firmwareVersion = cbase.AxiVersion.FpgaVersion.get()
        for seg in range(4):
            carrierId = [ cbase.SystemRegs.CarrierIdLow [seg].get(),
                          cbase.SystemRegs.CarrierIdHigh[seg].get() ]
            digitalId = [ 0, 0 ]
            analogId  = [ 0, 0 ]
            id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                              carrierId[0], carrierId[1],
                                                              digitalId[0], digitalId[1],
                                                              analogId [0], analogId [1])
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epix', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigMap[4*seg:4*seg+4].tolist(), 'UINT8')
            if trbit is not None:
                top.set('trbit'          , trbit[4*seg:4*seg+4], 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

#    for i in range(len(scfg)):
#        base['log'].write('--update-- {}\n'.format(i))
#        base['log'].write(result[i])

    return result

#
#  Check that ADC startup has completed successfully
#
def _checkADCs():
    cbase = base['cam']
    tmo = 0
    while True:
        time.sleep(0.001)
        if cbase.SystemRegs.AdcTestFailed.get()==1:
            print('Adc Test Failed - restarting!')
            cbase.SystemRegs.AdcReqStart.set(1)
            time.sleep(1.e-6)
            cbase.SystemRegs.AdcReqStart.set(0)
        else:
            tmo += 1
            if tmo > 1000:
                print('Adc Test Timedout')
                return 1
        if cbase.SystemRegs.AdcTestDone.get()==1:
            break
    print(f'Adc Test Done after {tmo} cycles')
    return 0

def _resetSequenceCount():
    cbase = base['cam']
    cbase.AcqCore.AcqCountReset.set(1)
    cbase.RdoutCore.SeqCountReset.set(1)
    time.sleep(1.e6)
    cbase.AcqCore.AcqCountReset.set(0)
    cbase.RdoutCore.SeqCountReset.set(0)

from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.cas.xpm_utils import timTxId
from .xpmmini import *
import rogue
import epix
import ePixQuad
import lcls2_pgp_pcie_apps
import time
import json
import os
import numpy as np
import IPython
from collections import deque
import surf.protocols.batcher  as batcher  # for Start/StopRun
import l2si_core               as l2si
import logging

base = None
pv = None
lane = 0
chan = None
group = None
ocfg = None
segids = None
seglist = [0,1,2,3,4]

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

def retry(cmd,val):
    itry=0
    while(True):
        try:
            cmd(val)
        except Exception as e:
            logging.warning(f'Try {itry} of {cmd}({val}) failed.')
            if itry < 3:
                itry+=1
                continue
            else:
                raise e
        break

#
#  Apply the configuration dictionary to the rogue registers
#
def apply_dict(pathbase,base,cfg):
    rogue_translate = {}
    rogue_translate['TriggerEventBuffer'] = f'TriggerEventBuffer[{lane}]'
    for i in range(16):
        rogue_translate[f'Epix10kaSaci{i}'] = f'Epix10kaSaci[{i}]'
    for i in range(3):
        rogue_translate[f'DbgOutSel{i}'] = f'DbgOutSel[{i}]'

    depth = 0
    my_queue  =  deque([[pathbase,depth,base,cfg]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        if(dict is type(configdb_node)):
            for i in configdb_node:
                k = rogue_translate[i] if i in rogue_translate else i
                try:
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[k],configdb_node[i]])
                except KeyError:
                    logging.warning('Lookup failed for node [{:}] in path [{:}]'.format(i,path))

        #  Apply
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path != pathbase ):
#            if False:
            if (('Saci' in path and 'PixelDummy' in path) or
                ('Saci3' in path and 'CompEn' in path) or
                ('Saci3' in path and 'Preamp' in path) or
                ('Saci3' in path and 'MonostPulser' in path) or
                ('Saci3' in path and 'PulserDac' in path)):
                logging.info(f'NOT setting {path} to {configdb_node}')
            else:
                logging.info(f'Setting {path} to {configdb_node}')
                retry(rogue_node.set,configdb_node)

#
#  Construct an asic pixel mask with square spacing
#
def pixel_mask_square(value0,value1,spacing,position):
    ny,nx=352,384;
    if position>=spacing**2:
        logging.error('position out of range')
        position=0;
    out=np.zeros((ny,nx),dtype=np.int)+value0
    position_x=position%spacing; position_y=position//spacing
    out[position_y::spacing,position_x::spacing]=value1
    return out

#
#  Initialize the rogue accessor
#
def epixquad_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M", verbose=0):
    global base
    global pv
    global lane
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.debug('epixquad_init')

    base = {}

    #  Configure the PCIe card first (timing, datavctap)
    if True:
        pbase = lcls2_pgp_pcie_apps.DevRoot(dev           =dev,
                                            enLclsI       =False,
                                            enLclsII      =True,
                                            yamlFileLclsI =None,
                                            yamlFileLclsII=None,
                                            startupMode   =True,
                                            standAloneMode=xpmpv is not None,
                                            pgp4          =True,
                                            dataVc        =0,
                                            pollEn        =False,
                                            initRead      =False)
        #dumpvars('pbase',pbase)

        pbase.__enter__()

        #  Disable flow control on the PGP lane at the PCIe end
#        getattr(pbase.DevPcie.Hsio,f'PgpMon[{lane}]').Ctrl.FlowControlDisable.set(1)

        # Open a new thread here
        if xpmpv is not None:
            pv = PVCtrls(xpmpv,pbase.DevPcie.Hsio.TimingRx.XpmMiniWrapper)
            pv.start()
        else:
            time.sleep(0.1)
        base['pci'] = pbase

    #  Connect to the camera
    cbase = ePixQuad.Top(dev=dev,hwType='datadev',lane=lane,pollEn=False,
                         enVcMask=0x2,enWriter=False,enPrbs=False)
    cbase.__enter__()
    base['cam'] = cbase

    firmwareVersion = cbase.AxiVersion.FpgaVersion.get()
    buildStamp = cbase.AxiVersion.BuildStamp.get()
    gitHash = cbase.AxiVersion.GitHash.get()
    print(f'firmwareVersion [{firmwareVersion:x}]')
    print(f'buildStamp      [{buildStamp}]')
    print(f'gitHash         [{gitHash:x}]')

    logging.info('epixquad_unconfig')
    epixquad_unconfig(base)

    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ModeSelEn.set(1)
    if timebase=="119M":
        logging.info('Using timebase 119M')
        base['clk_period'] = 1000/119.
        base['msg_period'] = 238
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(0)
    else:
        logging.info('Using timebase 186M')
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(1)
    #  To get the timing feedback link working
    pbase.DevPcie.Hsio.TimingRx.TimingPhyMonitor.TxPhyPllReset()
    time.sleep(1)
    #  Reset rx with the new reference
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.C_RxReset()
    time.sleep(2)
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.RxDown.set(0)

    epixquad_internal_trigger(base)
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
def epixquad_connectionInfo(base, alloc_json_str):

    if 'pci' in base:
        pbase = base['pci']
        rxId = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        logging.info('RxId {:x}'.format(rxId))
        txId = timTxId('epixquad')
        logging.info('TxId {:x}'.format(txId))
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    else:
        rxId = 0xffffffff


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
        triggerDelay   = int(rawStart/base['clk_period'] - partitionDelay*base['msg_period'])
        logging.debug('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        if triggerDelay < 0:
            logging.error('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
            raise ValueError('triggerDelay computes to < 0')

        d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.TriggerDelay']=triggerDelay

    if (hasUser and 'gate_ns' in cfg['user']):
        triggerWidth = int(cfg['user']['gate_ns']/10)
        if triggerWidth < 1:
            logging.error('triggerWidth {} ({} ns)'.format(triggerWidth,cfg['user']['gate_ns']))
            raise ValueError('triggerWidth computes to < 1')

        d[f'expert.EpixQuad.AcqCore.AsicAcqWidth']=triggerWidth

    if full:
        d[f'expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.Partition']=group

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
                d[f'expert.EpixQuad.Epix10kaSaci{i}.trbit'] = trbit
        logging.debug('pixel_map len {}'.format(len(a)))
        d['user.pixel_map'] = a
        pixel_map_changed = True

    update_config_entry(cfg,ocfg,d)

    return pixel_map_changed

#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writePixelMap=True):

    #  Disable internal triggers during configuration
    epixquad_external_trigger(base)

    # overwrite the low-level configuration parameters with calculations from the user configuration
    pbase = base['pci']
    if ('expert' in cfg and 'DevPcie' in cfg['expert']):
        apply_dict('pbase.DevPcie',pbase.DevPcie,cfg['expert']['DevPcie'])

    cbase = base['cam']

    #  Make list of enabled ASICs
    asics = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    #  Important that Asic IsEn is True while configuring and false when running
    for i in asics:
        logging.debug(f'Enabling ASIC {i}')
        saci = cbase.Epix10kaSaci[i]
        retry(saci.enable.set,True)
        retry(saci.IsEn.set,True)

    if ('expert' in cfg and 'EpixQuad' in cfg['expert']):
        epixQuad = cfg['expert']['EpixQuad'].copy()
        #  Add write protection word to upper range
        if 'AcqCore' in epixQuad and 'AsicRoClkT' in epixQuad['AcqCore']:
            epixQuad['AcqCore']['AsicRoClkT'] |= 0xaaaa0000
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

            logging.debug('SetAsicsMatrix complete')
        else:
            print('writePixelMap but no new map')
            logging.debug(cfg)

    #  Important that Asic IsEn is True while configuring and false when running
    for i in asics:
        saci = cbase.Epix10kaSaci[i]
        retry(saci.IsEn.set,False)
        retry(saci.enable.set,False)

    #  Enable triggers to continue monitoring
    epixquad_internal_trigger(base)

    logging.debug('config_expert complete')

def reset_counters(base):
    # Reset the timing counters
    base['pci'].DevPcie.Hsio.TimingRx.TimingFrameRx.countReset()

    # Reset the trigger counters
#    base['pci'].DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].countReset()
    getattr(base['pci'].DevPcie.Hsio.TimingRx.TriggerEventManager,f'TriggerEventBuffer[{lane}]').countReset()

    # Reset the Epix counters
    base['cam'].RdoutStreamMonitoring.countReset()

#
#  Modified version of DevRoot.StartRun touching only our lane
#
def startRun(pbase):
    logging.info('StartRun() executed')

    # Get devices
    eventBuilder = [getattr(pbase.DevPcie.Application,f'AppLane[{lane}]').EventBuilder]
    logging.info(f'startRun eventBuilder: {eventBuilder}')

    trigger      = [getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager,f'TriggerEventBuffer[{lane}]')]
    logging.info(f'startRun trigger: {trigger}')

    # Reset all counters
    pbase.CountReset()

    # Arm for data/trigger stream
    for devPtr in eventBuilder:
        devPtr.Blowoff.set(False)
        devPtr.SoftRst()

    # Turn on the triggering
    for devPtr in trigger:
        devPtr.MasterEnable.set(True)

    # Update the run state status variable
    pbase.RunState.set(True)

#
#  Modified version of DevRoot.StopRun touching only our lane
#
def stopRun(pbase):
    logging.info ('StopRun() executed')

    # Get devices
    eventBuilder = [getattr(pbase.DevPcie.Application,f'AppLane[{lane}]').EventBuilder]
    logging.info(f'stopRun eventBuilder: {eventBuilder}')

    trigger      = [getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager,f'TriggerEventBuffer[{lane}]')]
    logging.info(f'stopRun trigger: {trigger}')

    # Turn off the triggering
    for devPtr in trigger:
        devPtr.MasterEnable.set(False)

    # Flush the downstream data/trigger pipelines
    for devPtr in eventBuilder:
        devPtr.Blowoff.set(True)

    # Update the run state status variable
    pbase.RunState.set(False)


#
#  Called on Configure
#
def epixquad_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global ocfg
    global group
    global segids

    group = rog

    _checkADCs()

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
    #pbase.StartRun()
    startRun(pbase)

    #  Add some counter resets here
    reset_counters(base)

    #  Capture the firmware version to persist in the xtc
    cbase = base['cam']
    firmwareVersion = cbase.AxiVersion.FpgaVersion.get()

    ocfg = cfg

    #
    #  Create the segment configurations from parameters required for analysis
    #
    trbit = [ cfg['expert']['EpixQuad'][f'Epix10kaSaci{i}']['trbit'] for i in range(16)]

    topname = cfg['detName:RO'].split('_')

    scfg = {}
    segids = {}

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
        segids[seg] = id
        top = cdict()
        top.setAlg('config', [2,0,0])
        top.setInfo(detType='epix10ka', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
        top.set('asicPixelConfig', pixelConfigMap[4*seg:4*seg+4,:176].tolist(), 'UINT8')  # only the rows which have readable pixels
        top.set('trbit'          , trbit[4*seg:4*seg+4], 'UINT8')
        scfg[seg+1] = top.typed_json()

    result = []
    for i in seglist:
        logging.debug('json seg {}  detname {}'.format(i, scfg[i]['detName:RO']))
        result.append( json.dumps(scfg[i]) )

    return result

def epixquad_unconfig(base):
    pbase = base['pci']
    #pbase.StopRun()
    stopRun(pbase)
    return base

#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixquad_scan_keys(update):
    logging.debug('epixquad_scan_keys')
    global ocfg
    global base
    global segids

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
        trbit = [ cfg['expert']['EpixQuad'][f'Epix10kaSaci{i}']['trbit'] for i in range(16)]

        cbase = base['cam']
        for seg in range(4):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epix10ka', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigMap[4*seg:4*seg+4,:176].tolist(), 'UINT8')
            top.set('trbit'          , trbit[4*seg:4*seg+4], 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixquad_update(update):
    logging.debug('epixquad_update')
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
            trbit = [ cfg['expert']['EpixQuad'][f'Epix10kaSaci{i}']['trbit'] for i in range(16)]
        except:
            trbit = None

        cbase = base['cam']
        for seg in range(4):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epix10ka', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigMap[4*seg:4*seg+4,:176].tolist(), 'UINT8')
            if trbit is not None:
                top.set('trbit'          , trbit[4*seg:4*seg+4], 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

    logging.debug('update complete')

    return result

#
#  Check that ADC startup has completed successfully
#
def _checkADCs():

    epixquad_external_trigger(base)

    cbase = base['cam']
    tmo = 0
    while True:
        time.sleep(0.001)
        if cbase.SystemRegs.AdcTestFailed.get()==1:
            logging.warning('Adc Test Failed - restarting!')
            cbase.SystemRegs.AdcReqStart.set(1)
            time.sleep(1.e-6)
            cbase.SystemRegs.AdcReqStart.set(0)
        else:
            tmo += 1
            if tmo > 1000:
                logging.warning('Adc Test Timedout')
                return 1
        if cbase.SystemRegs.AdcTestDone.get()==1:
            break
    logging.debug(f'Adc Test Done after {tmo} cycles')

    epixquad_internal_trigger(base)

    return 0

def _resetSequenceCount():
    cbase = base['cam']
    cbase.AcqCore.AcqCountReset.set(1)
    cbase.RdoutCore.SeqCountReset.set(1)
    time.sleep(1.e6)
    cbase.AcqCore.AcqCountReset.set(0)
    cbase.RdoutCore.SeqCountReset.set(0)

def epixquad_external_trigger(base):
    cbase = base['cam']
    #  Switch to external triggering
    cbase.SystemRegs.AutoTrigEn.set(0)
    cbase.SystemRegs.TrigSrcSel.set(0)
    #  Enable frame readout
    cbase.RdoutCore.RdoutEn.set(1)

def epixquad_internal_trigger(base):
    cbase = base['cam']
    #  Disable frame readout
    cbase.RdoutCore.RdoutEn.set(0)
    #  Switch to internal triggering
    cbase.SystemRegs.TrigSrcSel.set(3)
    cbase.SystemRegs.AutoTrigEn.set(1)

def epixquad_enable(base):
    epixquad_external_trigger(base)

def epixquad_disable(base):
    time.sleep(0.005)  # Need to make sure readout of last event is complete
    epixquad_internal_trigger(base)

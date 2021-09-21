from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.cas.xpm_utils import timTxId
#from .xpmmini import *
import pyrogue as pr
import rogue
import lcls2_epix_hr_pcie
import epix_hr_single_10k
import epix_hr_core as epixHr
import ePixFpga as fpga
import time
import json
import os
import numpy as np
import IPython
from collections import deque
import logging

base = None
pv = None
#lane = 0  # An element consumes all 4 lanes
chan = None
group = None
ocfg = None
segids = None
seglist = [0,1]

elemRowsC = 146
elemRowsD = 144
elemCols  = 192

class Board(pr.Root):
    def __init__(self,dev='/dev/datadev_0',):
        super().__init__(name='ePixHr10kT',description='ePixHrGen1 board')
        self.dmaCtrlStreams = [None]
        self.dmaCtrlStreams[0] = rogue.hardware.axi.AxiStreamDma(dev,(0x100*0)+0,1)# Registers  

        # Create and Connect SRP to VC1 to send commands
        self._srp = rogue.protocols.srp.SrpV3()
        pr.streamConnectBiDir(self.dmaCtrlStreams[0],self._srp)

        self.add(epixHr.SysReg  (name='Core'  , memBase=self._srp, offset=0x00000000, sim=False, expand=False, pgpVersion=4,))
        self.add(fpga.EpixHR10kT(name='EpixHR', memBase=self._srp, offset=0x80000000, hidden=False, enabled=True))

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
    rogue_translate['TriggerEventBuffer'] = f'TriggerEventBuffer[0]'

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
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path is not pathbase ):
            if False:
                logging.info(f'NOT setting {path} to {configdb_node}')
            else:
                logging.info(f'Setting {path} to {configdb_node}')
                retry(rogue_node.set,configdb_node)

#
#  Construct an asic pixel mask with square spacing
#
def pixel_mask_square(value0,value1,spacing,position):
    ny,nx=288,384;
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
def epixhr2x2_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M"):
    global base
    global pv
    logging.debug('epixhr2x2_init')

    base = {}

    #  Configure the PCIe card first (timing, datavctap)
    if True:
        pbase = lcls2_epix_hr_pcie.DevRoot(dev           =dev,
                                           enLclsI       =False,
                                           enLclsII      =True,
                                           yamlFileLclsI =None,
                                           yamlFileLclsII=None,
                                           startupMode   =True,
                                           standAloneMode=xpmpv is not None,
                                           pgp4          =True,
                                           #dataVc        =0,
                                           pollEn        =False,
                                           initRead      =False,
                                           #numLanes      =4,
                                           pcieBoardType = 'Kcu1500')
        #dumpvars('pbase',pbase)

        pbase.__enter__()

        # Open a new thread here
#        if xpmpv is not None:
#            pv = PVCtrls(xpmpv,pbase.DevPcie.Hsio.TimingRx.XpmMiniWrapper)
#            pv.start()
#        else:
#            time.sleep(0.1)
        base['pci'] = pbase

    #  Connect to the camera
#    cbase = epix_hr_single_10k.ePixFpga.EpixHR10kT(dev=dev,hwType='datadev',lane=lane,pollEn=False,
#                                                   enVcMask=0x2,enWriter=False,enPrbs=False)
    cbase = Board(dev=dev)
    cbase.__enter__()
    base['cam'] = cbase

    logging.info('epixhr2x2_unconfig')
    epixhr2x2_unconfig(base)

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
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.RxDown.set(0)

    # configure internal ADC
#    cbase.InitHSADC()

    time.sleep(1)
    epixhr2x2_internal_trigger(base)
    return base

#
#  Set the PGP lane
#
def epixhr2x2_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

#
#  Set the local timing ID and fetch the remote timing ID
#
def epixhr2x2_connect(base):

    if 'pci' in base:
        pbase = base['pci']
        rxId = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        logging.debug('RxId {:x}'.format(rxId))
        txId = timTxId('epixhr2x2')
        logging.debug('TxId {:x}'.format(txId))
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    else:
        rxId = 0xffffffff


    epixhrid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixhrid

    return d

#
#  Helper function for calling underlying pyrogue interface
#
def intToBool(d,types,key):
    if isinstance(d[key],dict):
        for k,value in d[key].items():
            intToBool(d[key],types[key],k)
    elif types[key]=='boolEnum':
        d[key] = False if d[key]==0 else True
    
def dictToYaml(d,types,keys,dev,path,name):
    v = {'enable':True}
    for key in keys:
        v[key] = d[key]
        intToBool(v,types,key)
        v[key]['enable'] = True

    nd = {'ePixHr10kT':{'enable':True,'EpixHR':v}}
    yaml = pr.dataToYaml(nd)
    fn = path+name+'.yml'
    f = open(fn,'w')
    f.write(yaml)
    f.close()
    setattr(dev,'filename'+name,fn)

    #  Need to remove the enable field else Json2Xtc fails
    for key in keys:
        del d[key]['enable']

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

        d[f'expert.EpixHR.RegisterControl.AcqWidth1']=triggerWidth

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
            for i in range(4):
                d[f'expert.EpixHR.Hr10kTAsic{i}.trbit'] = trbit
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
    epixhr2x2_external_trigger(base)

    # overwrite the low-level configuration parameters with calculations from the user configuration
    pbase = base['pci']
    if ('expert' in cfg and 'DevPcie' in cfg['expert']):
        apply_dict('pbase.DevPcie',pbase.DevPcie,cfg['expert']['DevPcie'])


    cbase = base['cam']

    epixHR = None
    if ('expert' in cfg and 'EpixHR' in cfg['expert']):
        epixHR = cfg['expert']['EpixHR'].copy()

    #  Make list of enabled ASICs
    asics = []
    if 'user' in cfg and 'asic_enable' in cfg['user']:
        for i in range(4):
            if cfg['user']['asic_enable']&(1<<i):
                asics.append(i)
            else:
                # remove the ASIC configuration so we don't try it
                del epixHR['Hr10kTAsic{}'.format(i)]
    else:
        asics = [i for i in range(4)]

    #  Set the application event builder for the set of enabled asics
    m=3
    for i in asics:
        m = m | (4<<i)
    pbase.DevPcie.Application.EventBuilder.Bypass.set(0x3f^m)

    if epixHR is not None:
        # Work hard to use the underlying rogue interface
        # Translate config data to yaml files
        path = '/tmp/epixhr'
        epixHRTypes = cfg[':types:']['expert']['EpixHR']
        dictToYaml(epixHR,epixHRTypes,['MMCMRegisters'  ],cbase.EpixHR,path,'MMCM')
        dictToYaml(epixHR,epixHRTypes,['PowerSupply'    ],cbase.EpixHR,path,'PowerSupply')
        dictToYaml(epixHR,epixHRTypes,['RegisterControl'],cbase.EpixHR,path,'RegisterControl')
        for i in asics:
            dictToYaml(epixHR,epixHRTypes,['Hr10kTAsic{}'.format(i)],cbase.EpixHR,path,'ASIC{}'.format(i))
        dictToYaml(epixHR,epixHRTypes,['PacketRegisters{}'.format(i) for i in range(4)],cbase.EpixHR,path,'PacketReg')
        dictToYaml(epixHR,epixHRTypes,['TriggerRegisters'],cbase.EpixHR,path,'TriggerReg')

        arg = [1,0,0,0,0]
        for i in asics:
            arg[i+1] = 1
        print(f'Calling fnInitAsicScript(None,None,{arg})')
        cbase.EpixHR.fnInitAsicScript(None,None,arg)
                                      
    if writePixelMap:
        if 'user' in cfg and 'pixel_map' in cfg['user']:
            #  Write the pixel gain maps
            #  Would like to send a 3d array
            a = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
            pixelConfigMap = np.reshape(a,(4,146,192))
            if False:
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
                    saci = getattr(cbase.EpixHR,f'Hr10kTAsic{i}')
                    saci.CmdPrepForRead()
                    saci.PrepareMultiConfig()

                #  Set the whole ASIC to its most common value
                masic = {}
                for i in asics:
                    masic[i] = mode(pixelConfigMap[i])
                    saci = getattr(cbase.EpixHR,f'Hr10kTAsic{i}')
#                    saci.WriteMatrixData(masic)  # 0x4000 v 0x84000
                    for r in range(48):
                        saci.PrepareMultiConfig()
                        saci.ColCounter.set(r)
                        saci.WriteColData.set(masic[i])
                    saci.CmdPrepForRead()

                #  Now fix any pixels not at the common value
                for i in asics:
                    saci = getattr(cbase.EpixHR,f'Hr10kTAsic{i}')
                    for x in range(145):
                        for y in range(192):
                            if masic[i]==pixelConfigMap[i][x][y]:
                                continue
                            bankToWrite = int(y/48)
                            if bankToWrite==0:
                                colToWrite = 0x700 + y%48;
                            elif (bankToWrite == 1):
                                colToWrite = 0x680 + y%48;
                            elif (bankToWrite == 2):
                                colToWrite = 0x580 + y%48;
                            elif (bankToWrite == 3):
                                colToWrite = 0x380 + y%48;
                            else:
                                print('unexpected bank number')
                            saci.RowCounter.set(x) #6011
                            saci.ColCounter.set(colToWrite) #6013
                            saci.WritePixelData.set(int(pixelConfigMap[i][x][y])) #5000
                    saci.CmdPrepForRead()

            logging.debug('SetAsicsMatrix complete')
        else:
            print('writePixelMap but no new map')
            logging.debug(cfg)

    #  Enable triggers to continue monitoring
    epixhr2x2_internal_trigger(base)

    logging.debug('config_expert complete')

def reset_counters(base):
    # Reset the timing counters
    base['pci'].DevPcie.Hsio.TimingRx.TimingFrameRx.countReset()
    
    # Reset the trigger counters
    base['pci'].DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].countReset()

    # Reset the Epix counters
#    base['cam'].RdoutStreamMonitoring.countReset()

#
#  Called on Configure
#
def epixhr2x2_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global ocfg
    global group
    global segids

    group = rog

#    _checkADCs()

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    #  Translate user settings to the expert fields
    user_to_expert(base, cfg, full=True)

    #  Apply the expert settings to the device
    config_expert(base, cfg, writePixelMap=False)

    pbase = base['pci']
    pbase.StartRun()

    #  Add some counter resets here
    reset_counters(base)

    #  Capture the firmware version to persist in the xtc
    cbase = base['cam']
    firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()

    ocfg = cfg

    #
    #  Create the segment configurations from parameters required for analysis
    #
    trbit = [ cfg['expert']['EpixHR'][f'Hr10kTAsic{i}']['trbit'] for i in range(4)]

    topname = cfg['detName:RO'].split('_')

    scfg = {}
    segids = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'hw_'+topname[1]


    a = np.array(cfg['user']['pixel_map'],dtype=np.uint8)
    pixelConfigMap = np.reshape(a,(4,146,192))

    for seg in range(1):
        #  Construct the ID
#        carrierId = [ cbase.SystemRegs.CarrierIdLow [seg].get(),
#                      cbase.SystemRegs.CarrierIdHigh[seg].get() ]
        carrierId = [ 0, 0 ]
        digitalId = [ 0, 0 ]
        analogId  = [ 0, 0 ]
        id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'%(firmwareVersion,
                                                          carrierId[0], carrierId[1],
                                                          digitalId[0], digitalId[1],
                                                          analogId [0], analogId [1])
        segids[seg] = id
        top = cdict()
        top.setAlg('config', [2,0,0])
        top.setInfo(detType='epixhr2x2', detName=topname[0], detSegm=seg+int(topname[1]), detId=id, doc='No comment')
        top.set('asicPixelConfig', pixelConfigMap[seg:seg+1,:144].tolist(), 'UINT8')  # only the rows which have readable pixels
        top.set('trbit'          , trbit[seg:seg+1], 'UINT8')
        scfg[seg+1] = top.typed_json()

    result = []
    for i in seglist:
        logging.debug('json seg {}  detname {}'.format(i, scfg[i]['detName:RO']))
        result.append( json.dumps(scfg[i]) )

    return result

def epixhr2x2_unconfig(base):
    pbase = base['pci']
    pbase.StopRun()
    return base

#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixhr2x2_scan_keys(update):
    logging.debug('epixhr2x2_scan_keys')
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
        trbit = [ cfg['expert']['EpixHR'][f'Hr10kTAsic{i}']['trbit'] for i in range(16)]

        cbase = base['cam']
        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epixhr2x2', detName=topname[0], detSegm=int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigMap[seg:seg+1,:144].tolist(), 'UINT8')
            top.set('trbit'          , trbit[seg:seg+1], 'UINT8')
            scfg[seg+1] = top.typed_json()

    result = []
    for i in range(len(scfg)):
        result.append( json.dumps(scfg[i]) )

    return result

#
#  Return the set of configuration updates for a scan step
#
def epixhr2x2_update(update):
    logging.debug('epixhr2x2_update')
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
            trbit = [ cfg['expert']['EpixHR'][f'Hr10kTAsic{i}']['trbit'] for i in range(4)]
        except:
            trbit = None

        cbase = base['cam']
        for seg in range(1):
            id = segids[seg]
            top = cdict()
            top.setAlg('config', [2,0,0])
            top.setInfo(detType='epixhr2x2', detName=topname[0], detSegm=seg+int(topname[1]), detId=id, doc='No comment')
            top.set('asicPixelConfig', pixelConfigMap[seg:seg+1,:144].tolist(), 'UINT8')
            if trbit is not None:
                top.set('trbit'          , trbit[seg:seg+1], 'UINT8')
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

    epixhr2x2_external_trigger(base)

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

    epixhr2x2_internal_trigger(base)

    return 0

def _resetSequenceCount():
    cbase = base['cam']
    cbase.RegisterControl.ResetCounters.set(1)
    time.sleep(1.e6)
    cbase.RegisterControl.ResetCounters.set(0)

def epixhr2x2_external_trigger(base):
    cbase = base['cam'].EpixHR
    #  Switch to external triggering
    print('=== external triggering ===')
    cbase.TriggerRegisters.enable.set(1)
    cbase.TriggerRegisters.PgpTrigEn.set(1)
    cbase.TriggerRegisters.DaqTriggerEnable.set(1)
    cbase.TriggerRegisters.RunTriggerEnable.set(1)
    #  Enable frame readout
#    cbase.RdoutCore.RdoutEn.set(1)

def epixhr2x2_internal_trigger(base):
    cbase = base['cam'].EpixHR
    #  Disable frame readout 
    #  *** I don't know how to disable frame readout but still get monitoring
    #  *** So, we can't enable internal triggers, since it kills the EventBuilder
#    cbase.RdoutCore.RdoutEn.set(0)
    #  Switch to internal triggering
    print('=== internal triggering ===')
    cbase.TriggerRegisters.enable.set(1)
    cbase.TriggerRegisters.PgpTrigEn.set(0)

def epixhr2x2_enable(base):
    epixhr2x2_external_trigger(base)

def epixhr2x2_disable(base):
    epixhr2x2_internal_trigger(base)

#
#  Test standalone
#
if __name__ == "__main__":

    _base = epixhr2x2_init(None,dev='/dev/datadev_0')
    epixhr2x2_init_feb()
    epixhr2x2_connect(_base)

    db = 'https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/configDB'
    d = {'body':{'control':{'0':{'control_info':{'instrument':'asc',
                                                 'cfg_dbase' :db}}}}}
    _connect_str = json.dumps(d)
    epixhr2x2_config(_base,_connect_str,'BEAM','epixhr',0,1)

    epixhr2x2_enable(_base)

from psdaq.configdb.get_config import get_config
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

pv = None
lane = 0
chan = None

def dumpvars(prefix,c):
    print(prefix)
    for key,val in c.nodes.items():
        name = prefix+'.'+key
        dumpvars(name,val)

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
            rogue_node.set(configdb_node)


def epixquad_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None):

    global pv
    print('epixquad_init')
    print('pwd {}'.format(os.getcwd()))

    base = {}
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
    cbase = ePixQuad.Top(dev='/dev/datadev_1',hwType='datadev',lane=lane,pollEn=False)
    #dumpvars('cbase',cbase)
    cbase.__enter__()
    base['cam'] = cbase
    return base

def epixquad_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

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

def epixquad_config(base,connect_str,cfgtype,detname,detsegm,group):

    print('epixquad_config')

    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    if 'pci' in base:
        print('epixquad_config pbase')
        pbase = base['pci']
        # Clear the pipeline
        #getattr(pbase.DevPcie.Application,'AppLane[%d]'%lane).EventBuilder.Blowoff.set(True)

        # overwrite the low-level configuration parameters with calculations from the user configuration

        partitionDelay = getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
        rawStart       = cfg['user']['start_ns']
        triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        if triggerDelay < 0:
            print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
            raise ValueError('triggerDelay computes to < 0')

        cteb = cfg['expert']['DevPcie']['Hsio']['TimingRx']['TriggerEventManager'][f'TriggerEventBuffer[{lane}]']
        cteb['TriggerDelay'] = triggerDelay
        cteb['Partition'] = group

        apply_dict('pbase.DevPcie',pbase.DevPcie,cfg['expert']['DevPcie'])

    print('epixquad_config cbase')
    cbase = base['cam']

    #
    #  Configure the pixel gains
    #
    gain_mode = cfg['user']['gain_mode']
    if gain_mode==5:
        for i in range(16):
            cfg['expert']['EpixQuad'][f'Epix10kaSaci[{i}]']['trbit'] = cfg['user']['gain_map'][i][0][0]&1
        a = np.array(cfg['user']['gain_map'],dtype=np.uint8) & 0xc
        pixelConfigMap = np.reshape(a,(16,178,192))
    else:
        mapv  = (0xc,0xc,0x8,0x0,0x0)[gain_mode]
        trbit = (0x1,0x0,0x0,0x1,0x0)[gain_mode]
        # Now what?
        # Collect the mapv bits in a memArray and write like SaciConfigCore.setAsicsMatrix()
        # The trbit goes into the asic configuration
        a = np.zeros((16,178,192),dtype=np.uint8) | mapv
        print(f'a {a.shape} {a.dtype}')
        b = np.array(cfg['user']['gain_map'])
        print(f'b {b.shape} {b.dtype}')
        pixelConfigMap = a.copy()
        for i in range(16):
            cfg['expert']['EpixQuad'][f'Epix10kaSaci[{i}]']['trbit'] = trbit
            a[i] = a[i] | trbit
    cfg['user']['gain_map'] = a.reshape(-1).tolist()

    apply_dict('cbase',cbase,cfg['expert']['EpixQuad'])

    #  Write the pixel gain maps
    #  Would like to send a 3d array
    core = cbase.SaciConfigCore
    #core.SetAsicsMatrix(core,'setAsicsMatrixArray',pixelConfigMap)
    core.SetAsicsMatrix(json.dumps(pixelConfigMap.tolist()))
    print('SetAsicsMatrix complete')

    if 'pci' in base:
        #pbase.DevPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.HwEnable.set(True)
        #getattr(pbase.DevPcie.Hsio.TimingRx.TriggerEventManager,f'TriggerEventBuffer[{lane}]').MasterEnable.set(True)
        #getattr(pbase.DevPcie.Application,'AppLane[%d]'%lane).EventBuilder.Blowoff.set(False)
        pbase.StartRun()

    #  Capture the firmware version to persist in the xtc
    firmwareVersion = cbase.AxiVersion.FpgaVersion.get()
    #cfg['firmwareVersion:RO'] = cbase.AxiVersion.FpgaVersion.get()
    #cfg['firmwareBuild:RO'  ] = cbase.AxiVersion.BuildStamp.get()

    #return [json.dumps(cfg)]

    #
    #  Create the segment configurations from parameters required for analysis
    #

    trbit = [ cfg['expert']['EpixQuad'][f'Epix10kaSaci[{i}]']['trbit'] for i in range(16)]

    scfg = {}
    scfg[0] = cfg

    for seg in range(4):
        #  Construct the ID
        carrierId = [ cbase.SystemRegs.CarrierIdLow [seg].get(),
                      cbase.SystemRegs.CarrierIdHigh[seg].get() ]
        digitalId = [ 0, 0 ]
        analogId  = [ 0, 0 ]
        id = '%010d-%010d-%010d-%010d-%010d-%010d-%010d'.format(firmwareVersion,
                                                                carrierId[0], carrierId[1],
                                                                digitalId[0], digitalId[1],
                                                                analogId [0], analogId [1])
        top = cdict()
        top.setAlg('config', [2,0,0])
        topname = cfg['detName:RO'].split('_')
        top.setInfo(detType='epix', detName=topname[0], detSegm=seg+4*int(topname[1]), detId=id, doc='No comment')
        top.set('asicPixelConfig', pixelConfigMap[4*seg:4*seg+4].tolist(), 'UINT8')
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

def epixquad_scan_keys(update):
    print('epixquad_scan_keys')
    cfg = {}
    return json.dumps(cfg)

def epixquad_update(update):
    print('epixquad_update')
    cfg = {}
    return json.dumps(cfg)


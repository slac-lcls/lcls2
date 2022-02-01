from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.cas.xpm_utils import timTxId
import pyrogue as pr
import rogue
import lcls2_epix_hr_pcie
#import ePixFpga as fpga
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
#cposeglist
#seglist = [0,1]
seglist = [0]

elemRowsC = 146
elemRowsD = 144
elemCols  = 192

def gain_mode_map(gain_mode):
    mapv  = (0xc,0xc,0x8,0x0,0x0)[gain_mode] # H/M/L/AHL/AML
    trbit = (0x1,0x0,0x0,0x1,0x0)[gain_mode]
    return (mapv,trbit)

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

def _dict_compare(d1,d2,path):
    for k in d1.keys():
        if k in d2.keys():
            if d1[k] is dict:
                _dict_compare(d1[k],d2[k],path+'.'+k)
            elif (d1[k] != d2[k]):
                print(f'key[{k}] d1[{d1[k]}] != d2[{d2[k]}]')
        else:
            print(f'key[{k}] not in d1')
    for k in d2.keys():
        if k not in d1.keys():
            print(f'key[{k}] not in d2')

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
#  Scramble the user element pixel array into the native asic orientation
#
#
#    A1   |   A3       (A1,A3) rotated 180deg
# --------+--------
#    A0   |   A2
#
def user_to_rogue(a):
    v = a.reshape((elemRowsD*2,elemCols*2))
    s = np.zeros((4,elemRowsC,elemCols),dtype=np.uint8)
    s[0,:elemRowsD] = v[elemRowsD:,:elemCols]
    s[2,:elemRowsD] = v[elemRowsD:,elemCols:]
    vf = np.flip(v)
    s[1,:elemRowsD] = vf[elemRowsD:,elemCols:]
    s[3,:elemRowsD] = vf[elemRowsD:,:elemCols]

    return s

#
#  Initialize the rogue accessor
#
def epix100_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M",verbosity=0):
    global base
    global pv
    print('*** here in epix100_init')

#    logging.getLogger().setLevel(40-10*verbosity)
    logging.debug('epix100_init')

    base = {}

    #  Configure the PCIe card first (timing, datavctap)
    #if True:
    print('**** cpo hack out kcu1500')
    if False:
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

        pbase.__enter__()

        base['pci'] = pbase
        pbase.DevPcie.Application.EventBuilder.Blowoff.set(True)
        time.sleep(1)
        pbase.DevPcie.Application.EventBuilder.Blowoff.set(False)

    print('*** cpo hack early return')
    return base # cpo hack

    #  Connect to the camera
#    cbase = epix_hr_single_10k.ePixFpga.EpixHR10kT(dev=dev,hwType='datadev',lane=lane,pollEn=False,
#                                                   enVcMask=0x2,enWriter=False,enPrbs=False)
    cbase = Board(dev=dev)
    cbase.__enter__()
    base['cam'] = cbase

    base['bypass'] = 0x3f

    #  Enable the environmental monitoring
    cbase.EpixHR.SlowAdcRegisters.enable.set(1)
    cbase.EpixHR.SlowAdcRegisters.StreamPeriod.set(100000000)  # 1Hz
    cbase.EpixHR.SlowAdcRegisters.StreamEn.set(1)
    cbase.EpixHR.SlowAdcRegisters.enable.set(0)

    logging.info('epix100_unconfig')
    epix100_unconfig(base)

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
    epix100_internal_trigger(base)
    return base

#
#  Set the PGP lane
#
def epix100_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

#
#  Set the local timing ID and fetch the remote timing ID
#
def epix100_connect(base):

    if 'pci' in base:
        pbase = base['pci']
        rxId = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        logging.debug('RxId {:x}'.format(rxId))
        txId = timTxId('epix100')
        logging.debug('TxId {:x}'.format(txId))
        pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
        #print('*** cpohack partition/readout group to 4')
        #pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition=4
    else:
        print('*** cpohack rxid')
        #rxId = 0xffffffff
        rxId = 0xfffffffe


    epixhrid = '-'

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixhrid

    return d

#
#  Called on Configure
#
def epix100_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global ocfg
    global group
    global segids

    group = rog

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    #  Apply the expert settings to the device
    if 'pci' in base:
        pbase = base['pci']
        pbase.StopRun()
        time.sleep(0.01)
        
        pbase.StartRun()

    #  Capture the firmware version to persist in the xtc
    #cbase = base['cam']
    #firmwareVersion = cbase.Core.AxiVersion.FpgaVersion.get()
    firmwareVersion = 567

    ocfg = cfg

    topname = cfg['detName:RO'].split('_')

    scfg = {}
    segids = {}

    #  Rename the complete config detector
    scfg[0] = cfg.copy()
    scfg[0]['detName:RO'] = topname[0]+'_'+topname[1]


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
        top.setInfo(detType='epix100', detName=topname[0], detSegm=seg+int(topname[1]), detId=id, doc='No comment')
        scfg[seg+1] = top.typed_json()

    result = []
    for i in seglist:
        logging.debug('json seg {}  detname {}'.format(i, scfg[i]['detName:RO']))
        result.append( json.dumps(scfg[i]) )

    print('****',result)

    return result

def epix100_unconfig(base):
    pbase = base['pci']
    pbase.StopRun()
    return base

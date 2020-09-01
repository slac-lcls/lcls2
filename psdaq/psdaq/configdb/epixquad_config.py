from psdaq.configdb.get_config import get_config
from psdaq.cas.xpm_utils import timTxId
from .xpmmini import *
from .epixquad import *
import rogue
import epix
import time
import json
import IPython
from collections import deque

pv = None

#
#  Need a reorganized epixQuad.Top (a la cameralink-gateway)
#    Factor into devTarget (KCU, PgpG4,...) and devRoot
#      devTarget(pr.Device) contains register description on host card
#      devRoot(shared.Root):
#        maps PCIe
#        instanciates devTarget
#        creates streams
#        instanciates and connects feb

class EpixRoot(lcls2_pgp_pcie_apps.DevRoot):
    def __init__(self):

        self.devTarget = lcls2_pgp_pcie_apps.Kcu1500
        numLanes  = 4

        myargs = { 'dev'         : '/dev/datadev_0',
                   'pollEn'      : False,
                   'initRead'    : True,
                   'pgp3'        : True,
                   'dataVc'      : 0,
                   'enLclsI'     : False,
                   'enLclsII'    : True,
                   'startupMode' : True,
                   'standAloneMode' : False,
                   'numLanes'    : numLanes,
                   'devTarget'   : self.devTarget,}

        super().__init__(**kwargs)

        lane = 0
        dev  = myargs['dev']
        self.pgpVc0 = rogue.hardware.axi.AxiStreamDma(dev,256*lane+0,True) # Data & cmds
        self.pgpVc1 = rogue.hardware.axi.AxiStreamDma(dev,256*lane+1,True) # Registers for ePix board
        self.pgpVc2 = rogue.hardware.axi.AxiStreamDma(dev,256*lane+2,True) # PseudoScope
        self.pgpVc3 = rogue.hardware.axi.AxiStreamDma(dev,256*lane+3,True) # Monitoring (Slow ADC)        

        # Connect the SRPv3 to PGPv3.VC[0]
        memMap = rogue.protocols.srp.SrpV3()                
        pr.streamConnectBiDir(self.pgpVc1, memMap)             
      
        #pyrogue.streamConnect(self.pgpVc0, dataWriter.getChannel(0x1))
        # Add pseudoscope to file writer
        #pyrogue.streamConnect(self.pgpVc2, dataWriter.getChannel(0x2))
        #pyrogue.streamConnect(self.pgpVc3, dataWriter.getChannel(0x3))
        
        #cmdVc1 = rogue.protocols.srp.Cmd()
        #pyrogue.streamConnect(cmdVc1, self.pgpVc0)
        cmdVc3 = rogue.protocols.srp.Cmd()
        pyrogue.streamConnect(cmdVc3, self.pgpVc3)

        self.add(EpixQuad(name    = 'EpixQuad',
                          memMap  = memMap))

    @self.command()
    def MonStrEnable():
        cmdVc3.sendCmd(1, 0)
      
    @self.command()
    def MonStrDisable():
        cmdVc3.sendCmd(0, 0)
      
      


def epixquad_init(arg,xpmpv=None):

    global pv
    print('epixquad_init')

    # in older versions we didn't have to use the "with" statement
    # but now the register accesses don't seem to work without it -cpo
    base = EpixRoot()
    base.__enter__()

    # Open a new thread here
    if xpmpv is not None:
        base.DevPcie.Hsio.TimingRx.ConfigureXpmMini()
        pv = PVCtrls(xpmpv,base.DevPcie.Hsio.TimingRx.XpmMiniWrapper)
        pv.start()
    else:
        base.DevPcie.Hsio.TimingRx.ConfigLclsTimingV2()
        time.sleep(0.1)

    return base

def epixquad_connect(base):

    txId = timTxId('epixquad')

    rxId = base.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    base.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)

    print('rxId {:x}'.format(rxId))

    # fetch the serial number
    # SystemRegs->CarrierIdLow/High[0:3]
    # {"DigitalCardId0"},
    # {"DigitalCardId1"},
    # {"AnalogCardId0"},
    # {"AnalogCardId1"},
    # {"CarrierId0"},
    # {"CarrierId1"}
    # 0x30-0x33,0x3b-0x3c

    epixquadid = 0

    d = {}
    d['paddr'] = rxId
    d['serno'] = epixquadid

    return d

def epixquad_config(base,connect_str,cfgtype,detname,detsegm,group):

    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    # Clear the pipeline
    getattr(base.DevPcie.Application,'AppLane[%d]'%lane).EventBuilder.Blowoff.set(True)

    # overwrite the low-level configuration parameters with calculations from the user configuration

    partitionDelay = getattr(base.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
    rawStart       = cfg['user']['start_ns']
    triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
    print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
    if triggerDelay < 0:
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        raise ValueError('triggerDelay computes to < 0')

    cfg['expert']['DevPcie']['Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer[0]']['TriggerDelay'] = triggerDelay
    cfg['expert']['DevPcie']['Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer[0]']['Partition'] = group

    depth = 0
    path  = 'base'
    my_queue  =  deque([[path,depth,base,cfg['expert']]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        if(dict is type(configdb_node)):
            for i in configdb_node:
                try:
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[i],configdb_node[i]])
                except KeyError:
                    print('Lookup failed for node [{:}] in path [{:}]'.format(i,path))

        #  Apply
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path is not 'base' ):
            rogue_node.set(configdb_node)

    #
    #  Configure the pixel gains
    #
    gain_mode = cfg['user']['gain_mode']
    if gain_mode==5:
        #  load from file
        base.EpixQuad.SaciConfigCore.setAsicsMatrix(None,None,cfg['user']['gain_file'])
    else:
        mapv  = (0xc,0xc,0x8,0x0,0x0)[gain_mode]
        trbit = (0x1,0x0,0x0,0x1,0x0)[gain_mode]


    base.DevPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.HwEnable.set(True)
    base.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(True)
    getattr(base.DevPcie.Application,'AppLane[%d]'%lane).EventBuilder.Blowoff.set(False)

    #  Capture the firmware version to persist in the xtc
    #cfg['firmwareVersion'] = base.ClinkPcie.AxiPcieCore.AxiVersion.FpgaVersion.get()
    #cfg['firmwareBuild'  ] = base.ClinkPcie.AxiPcieCore.AxiVersion.BuildStamp.get()

    #base.StartRun()

    return json.dumps(cfg)

def epixquad_unconfig(base):
    base.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(False)
    return base

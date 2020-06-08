from psdaq.configdb.get_config import get_config
from psdaq.cas.xpm_utils import timTxId
from .xpmmini import *
import rogue
import lcls2_timetool
import json
import IPython

from collections import deque

pv = None

def tt_init(arg,xpmpv=None):

    myargs = { 'dev'         : '/dev/datadev_0',
               'pgp3'        : False,
               'pollEn'      : False,
               'initRead'    : False,
               'dataCapture' : False,
               'dataDebug'   : False,}

    cl = lcls2_timetool.TimeToolKcu1500Root(**myargs)
    cl.__enter__()

    if(cl.TimeToolKcu1500.Kcu1500Hsio.PgpMon[0].RxRemLinkReady.get() != 1):
        raise ValueError(f'PGP Link is down' )

    # Open a new thread here
    if xpmpv is not None:
        cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.ConfigureXpmMini()
        pv = PVCtrls(xpmpv,cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.XpmMiniWrapper)
        pv.start()
    else:
        print('XpmMini')
        cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.ConfigureXpmMini()
        cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.ConfigLclsTimingV2()
        time.sleep(0.1)

    return cl

def tt_connect(cl):

    txId = timTxId('timetool')

    rxId = cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
        
    d = {}
    d['paddr'] = rxId
    return d

def tt_config(cl,connect_str,cfgtype,detname,detsegm,group):

    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    # TimeToolKcu1500Root.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId
    partitionDelay = getattr(cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
    rawStart       = cfg['user']['start_ns']
    triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
    print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
    if triggerDelay < 0:
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        raise ValueError('triggerDelay computes to < 0')
            
    cfg['cl']['TimeToolKcu1500']['Kcu1500Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer0']['TriggerDelay'] = triggerDelay
    cfg['cl']['TimeToolKcu1500']['Kcu1500Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer0']['Partition'] = group
        
    cl.ClinkFeb[0].ClinkTop.Ch[0].UartPiranha4.SendEscape()

    depth = 0
    path  = 'cl'
    my_queue  =  deque([[path,depth,cl,cfg['cl']]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    kludge_dict = {"TriggerEventBuffer0":"TriggerEventBuffer[0]","AppLane0":"AppLane[0]","ClinkFeb0":"ClinkFeb[0]","Ch0":"Ch[0]", "ROI0":"ROI[0]","ROI1":"ROI[1]","SAD0":"SAD[0]","SAD1":"SAD[1]","SAD2":"SAD[2]"}
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        if(dict is type(configdb_node)):
            for i in configdb_node:
                if i in kludge_dict:
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[kludge_dict[i]],configdb_node[i]])
                else:
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[i],configdb_node[i]])
        
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path is not 'cl' ):

            #  All FIR parameters are stored in configdb as hex strings (I don't know why)
            if ".FIR." in path:
                print(path+", rogue value = "+str(hex(rogue_node.get()))+", daq config database = " +str(configdb_node))
                rogue_node.set(int(str(configdb_node),16))
            else:
                rogue_node.set(configdb_node)
    
    #cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.XpmMiniWrapper.XpmMini.HwEnable.set(True)
    cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(True)

    #  Capture the firmware version to persist in the xtc
    cfg['firmwareVersion'] = cl.TimeToolKcu1500.AxiPcieCore.AxiVersion.FpgaVersion.get()
    cfg['firmwareBuild'  ] = cl.TimeToolKcu1500.AxiPcieCore.AxiVersion.BuildStamp.get()

    return json.dumps(cfg)

def tt_unconfig(cl):
    cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(False)
    

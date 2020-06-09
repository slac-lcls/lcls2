from psdaq.configdb.get_config import get_config
from psdaq.cas.xpm_utils import timTxId
from .xpmmini import *
import rogue
import cameralink_gateway
import time
import json
import IPython
from collections import deque

pv = None

#FEB parameters
lane = 0
chan = 0

def opal_init(arg,xpmpv=None):

    global pv
    print('opal_init')

    myargs = { 'dev'         : '/dev/datadev_0',
               'pollEn'      : False,
               'initRead'    : True,
               'camType'     : ['Opal1000'],
               'dataDebug'   : False,}

    # in older versions we didn't have to use the "with" statement
    # but now the register accesses don't seem to work without it -cpo
    cl = cameralink_gateway.ClinkDevRoot(**myargs)
    cl.__enter__()

    # Open a new thread here
    if xpmpv is not None:
        cl.ClinkPcie.Hsio.TimingRx.ConfigureXpmMini()
        pv = PVCtrls(xpmpv,cl.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper)
        pv.start()
    else:
        cl.ClinkPcie.Hsio.TimingRx.ConfigLclsTimingV2()
        time.sleep(0.1)

    return cl

def opal_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

def opal_connect(cl):

    txId = timTxId('opal')

    rxId = cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)

    print('rxId {:x}'.format(rxId))

    # initialize the serial link
    uart = getattr(getattr(cl,'ClinkFeb[%d]'%lane).ClinkTop,'Ch[%d]'%chan)
    uart.BaudRate.set(57600)
    uart.SerThrottle.set(10000)
    time.sleep(0.10)

    # @ID? returns OPAL-1000m/Q S/N:xxxxxxxx
    uart = getattr(getattr(cl,'ClinkFeb[%d]'%lane).ClinkTop,'Ch[%d]'%chan).UartOpal1000
    try:
        v = uart.ID.get()
        uart.ID.set(str(int(v)+1))
    except:
        uart.ID.set('0')
    time.sleep(0.10)
    opalid = uart._rx._last
    print('opalid {:}'.format(opalid))

    cl.StopRun()

    d = {}
    d['paddr'] = rxId
    d['model'] = opalid.split('-')[1].split('/')[0]
    d['serno'] = opalid.split(':')[1]

    return d

def opal_config(cl,connect_str,cfgtype,detname,detsegm,group):

    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    lane = 0
    chan = 0

    if(cl.ClinkPcie.Hsio.PgpMon[0].RxRemLinkReady.get() != 1):
        raise ValueError(f'PGP Link is down' )

    # drain any data in the event pipeline
    getattr(cl.ClinkPcie.Application,'AppLane[%d]'%lane).EventBuilder.Blowoff.set(True)
    getattr(getattr(cl,'ClinkFeb[%d]'%lane).ClinkTop,'Ch[%d]'%chan).Blowoff.set(True)

    # overwrite the low-level configuration parameters with calculations from the user configuration

    partitionDelay = getattr(cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
    rawStart       = cfg['user']['start_ns']
    triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
    print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
    if triggerDelay < 0:
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        raise ValueError('triggerDelay computes to < 0')

    cfg['expert']['ClinkPcie']['Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer[0]']['TriggerDelay'] = triggerDelay
    cfg['expert']['ClinkPcie']['Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer[0]']['Partition'] = group

    gate = cfg['user']['gate_ns']
    cfg['expert']['ClinkFeb[0]']['TrigCtrl[0]']['TrigPulseWidth'] = gate*0.001
    cfg['expert']['ClinkFeb[0]']['ClinkTop']['Ch[0]']['UartOpal1000']['BL']   = cfg['user']['black_level']
    cfg['expert']['ClinkFeb[0]']['ClinkTop']['Ch[0]']['UartOpal1000']['VBIN'] = cfg['user']['vertical_bin']

    #  set bool parameters
    cfg['expert']['ClinkFeb[0]']['TrigCtrl[0]']['EnableTrig'] = True
    cfg['expert']['ClinkFeb[0]']['TrigCtrl[0]']['InvCC'] = False
    cfg['expert']['ClinkFeb[0]']['ClinkTop']['Ch[0]']['DataEn'] = True
    cfg['expert']['ClinkFeb[0]']['ClinkTop']['Ch[0]']['Blowoff'] = False

    #  trigger polarity inversion for firmware version
    uart = getattr(getattr(cl,'ClinkFeb[%d]'%lane).ClinkTop,'Ch[%d]'%chan).UartOpal1000
    try:
        v = uart.BS.get()
        uart.BS.set(str(int(v)+1))
    except:
        uart.BS.set('0')
    time.sleep(0.10)
    fwver = uart._rx._last.split(';')
    if len(fwver)>2:
        major,minor = fwver[2].split('.')[:2]
        if int(major)<1 or int(minor)<20:
            print('Normal polarity')
            cfg['expert']['ClinkFeb[0]']['ClinkTop']['Ch[0]']['UartOpal1000']['CCE[1]'] = 0 
        else:
            print('Inverted polarity')
            cfg['expert']['ClinkFeb[0]']['ClinkTop']['Ch[0]']['UartOpal1000']['CCE[1]'] = 1  # invert polarity 

    depth = 0
    path  = 'cl'
    my_queue  =  deque([[path,depth,cl,cfg['expert']]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        #  Replace configdb lane and febch for the physical values
        if(dict is type(configdb_node)):
            for i in configdb_node:
                #  Substitute proper pgp lane or feb channel
                if i == 'ClinkFeb[0]':
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes['ClinkFeb[%d]'%lane],configdb_node[i]])
                elif i == 'Ch[0]':
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes['Ch[%d]'%chan],configdb_node[i]])
                else:
                    try:
                        my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[i],configdb_node[i]])
                    except KeyError:
                        print('Lookup failed for node [{:}] in path [{:}]'.format(i,path))

        #  Apply
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path is not 'cl' ):
            rogue_node.set(configdb_node)


    cl.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.HwEnable.set(True)
    cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(True)
    getattr(cl.ClinkPcie.Application,'AppLane[%d]'%lane).EventBuilder.Blowoff.set(False)

    #  Capture the firmware version to persist in the xtc
    cfg['firmwareVersion'] = cl.ClinkPcie.AxiPcieCore.AxiVersion.FpgaVersion.get()
    cfg['firmwareBuild'  ] = cl.ClinkPcie.AxiPcieCore.AxiVersion.BuildStamp.get()

    cl.StartRun()

    return json.dumps(cfg)

def opal_unconfig(cl):
    cl.StopRun()

#    cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(False)
    return cl

from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.xpmmini import *
from psdaq.cas.xpm_utils import timTxId
import rogue
import cameralink_gateway
import time
import json
import IPython
from collections import deque

cl = None
pv = None
lm = 1

#FEB parameters
lane = 0
chan = 0
ocfg = None
group = None

def dict_compare(new,curr,result):
    for k in new.keys():
        if dict is type(curr[k]):
            resultk = {}
            dict_compare(new[k],curr[k],resultk)
            if resultk:
                result[k] = resultk
        else:
            if new[k]==curr[k]: 
                pass
            else:
                result[k] = new[k]

def opal_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None):

    global pv
    global cl
    global lm
    print('opal_init')

    myargs = { 'dev'         : dev,
               'pollEn'      : False,
               'initRead'    : True,
               'camType'     : ['Opal1000'],
               'dataDebug'   : False,
               'enLclsII'    : True,
    }

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

    # the opal seems to intermittently lose lock back to the XPM
    # and empirically this fixes it.  not sure if we need the sleep - cpo
    cl.ClinkPcie.Hsio.TimingRx.TimingPhyMonitor.TxPhyReset()
    time.sleep(0.1)

    lm=lanemask
    return cl

def opal_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)

def opal_connect(cl):
    global lane
    global chan

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

def user_to_expert(cl, cfg, full=False):
    global group

    d = {}
    hasUser = 'user' in cfg
    if (hasUser and 'start_ns' in cfg['user']):
        partitionDelay = getattr(cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
        rawStart       = cfg['user']['start_ns']
        triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        if triggerDelay < 0:
            print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
            raise ValueError('triggerDelay computes to < 0')

        d['expert.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.TriggerDelay']=triggerDelay

    if full:
        d['expert.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.Partition']=group

    if (hasUser and 'gate_ns' in cfg['user']):
        gate = cfg['user']['gate_ns']
        if gate > 160000:
            print('gate_ns {:} may cause errors.  Please use a smaller gate'.format(gate));
            raise ValueError('gate_ns > 160000')
        d['expert.ClinkFeb.TrigCtrl.TrigPulseWidth']=gate*0.001

    if (hasUser and 'black_level' in cfg['user']):
        d['expert.ClinkFeb.ClinkTop.ClinkCh.UartOpal1000.BL']=cfg['user']['black_level']

    if (hasUser and 'vertical_bin' in cfg['user']):
        d['expert.ClinkFeb.ClinkTop.ClinkCh.UartOpal1000.VBIN']=cfg['user']['vertical_bin']

    update_config_entry(cfg,ocfg,d)

def config_expert(cl, cfg):
    global lane
    global chan

    # translate legal Python names to Rogue names
    rogue_translate = {'ClinkFeb'          :'ClinkFeb[%d]'%lane,
                       'ClinkCh'           :'Ch[%d]'%chan,
                       'TriggerEventBuffer':'TriggerEventBuffer[%d]'%lane,
                       'TrigCtrl'          :'TrigCtrl[%d]'%chan,
                       'PllConfig0'        :'PllConfig[0]',
                       'PllConfig1'        :'PllConfig[1]',
                       'PllConfig2'        :'PllConfig[2]',
                       'Red'               :'WB[0]',
                       'Green'             :'WB[1]',
                       'Blue'              :'WB[2]'}

    depth = 0
    path  = 'cl'
    my_queue  =  deque([[path,depth,cl,cfg]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        #  Replace configdb lane and febch for the physical values
        if(dict is type(configdb_node)):
            for i in configdb_node:
                if i in rogue_translate:
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[rogue_translate[i]],configdb_node[i]])
                else:
                    try:
                        my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[i],configdb_node[i]])
                    except KeyError:
                        print('Lookup failed for node [{:}] in path [{:}]'.format(i,path))

        #  Apply
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path is not 'cl' ):
            rogue_node.set(configdb_node)

    #  Parameters like black-level need time to take affect (100ms?)

#  Apply the full configuration
def opal_config(cl,connect_str,cfgtype,detname,detsegm,grp):
    global ocfg
    global group
    global lane
    global chan
    group = grp

    appLane  = 'AppLane[%d]'%lane
    clinkFeb = 'ClinkFeb[%d]'%lane
    clinkCh  = 'Ch[%d]'%chan

    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    if(cl.ClinkPcie.Hsio.PgpMon[0].RxRemLinkReady.get() != 1):
        raise ValueError(f'PGP Link is down' )

    # drain any data in the event pipeline
    getattr(cl.ClinkPcie.Application,appLane).EventBuilder.Blowoff.set(True)
    getattr(getattr(cl,clinkFeb).ClinkTop,clinkCh).Blowoff.set(True)

    #  set bool parameters
    cfg['expert']['ClinkFeb']['TrigCtrl']['EnableTrig'] = True
    cfg['expert']['ClinkFeb']['TrigCtrl']['InvCC'] = False
    cfg['expert']['ClinkFeb']['ClinkTop']['ClinkCh']['DataEn'] = True

    #  trigger polarity inversion for firmware version
    uart = getattr(getattr(cl,clinkFeb).ClinkTop,clinkCh).UartOpal1000
    try:
        v = uart.BS.get()
        uart.BS.set(str(int(v)+1))
    except:
        uart.BS.set('0')
    time.sleep(0.10)
    fwver = uart._rx._last.split(';')
    if len(fwver)>2:
        major,minor = fwver[2].split('.')[:2]
        # CCE is a special command in the rogue surf. it waits for
        # both CCE[0] and CCE[1] to be filled in before transmitting.
        # a possible issue: when we do a second configure, the
        # fields will be non-empty, so we think we will do two
        # uart writes of the same value.  not ideal, but should be ok.
        # we inherited this information about firmware version
        # numbers from LCLS1, but we are not confident it is solid.
        # for a given camera, the exposure time should be checked
        # by looking at the second set of 4 pixels of an image after
        # a long time between triggers which have an exposure time
        # measurement.  CCE[1] is the "polarity" portion of CCE:

        # cpo: somehow this logic is wrong because this response
        # from a camera's BS command @"1.10;1.11;1.11 wanted
        # normal polarity, but the exposure time (after a pause)
        # was very large. So hardwire inverted-polarity for now.

        #if int(major)<1 or (int(major)==1 and int(minor)<20):
        #    print('Normal polarity')
        #    getattr(uart,'CCE[1]').set(0)
        #else:
        #    print('Inverted polarity')
        #    getattr(uart,'CCE[1]').set(1)

        print('Inverted polarity')
        getattr(uart,'CCE[1]').set(1)

    # CCE[0] is the "trigger input source" portion of CCE.
    getattr(uart,'CCE[0]').set(0)  # trigger on CC1
    uart.MO.set(1)  # set to triggered mode

    user_to_expert(cl,cfg,full=True)

    config_expert(cl,cfg['expert'])

    cl.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.HwEnable.set(True)
    cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(True)
    getattr(getattr(cl,clinkFeb).ClinkTop,clinkCh).Blowoff.set(False)
    getattr(cl.ClinkPcie.Application,appLane).EventBuilder.Blowoff.set(False)

    #  Capture the firmware version to persist in the xtc
    cfg['firmwareVersion'] = cl.ClinkPcie.AxiPcieCore.AxiVersion.FpgaVersion.get()
    cfg['firmwareBuild'  ] = cl.ClinkPcie.AxiPcieCore.AxiVersion.BuildStamp.get()

    cl.StartRun()

    ocfg = cfg
    return json.dumps(cfg)

    #ncfg = cfg.copy()
    #del ncfg['expert']['ClinkFeb']
    #return json.dumps(ncfg)


def opal_scan_keys(update):
    global ocfg
    global cl
    #  extract updates
    cfg = {}
    copy_reconfig_keys(cfg,ocfg, json.loads(update))
    #  Apply group
    user_to_expert(cl,cfg,full=False)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def opal_update(update):
    global ocfg
    global cl
    #  extract updates
    cfg = {}
    update_config_entry(cfg,ocfg, json.loads(update))
    #  Apply group
    user_to_expert(cl,cfg,full=False)
    #  Apply config
    config_expert(cl, cfg['expert'])
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def opal_unconfig(cl):
    cl.StopRun()

    return cl

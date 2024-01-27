from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.xpmmini import *
from psdaq.configdb.barrier import Barrier
from psdaq.cas.xpm_utils import timTxId
import os
import socket
import rogue
import cameralink_gateway
import time
import json
import IPython
from collections import deque
import logging
import weakref

cl = None
pv = None
xpmpv_global = None
barrier_global = Barrier()
lm = 1

#FEB parameters
lane = 0
chan = 0
ocfg = None
group = None

def supervisor_info(connect_json):
    nworker = 0
    supervisor=None
    mypid = os.getpid()
    myhostname = socket.gethostname()
    for drp in connect_json['body']['drp'].values():
        proc_info = drp['proc_info']
        host = proc_info['host']
        pid = proc_info['pid']
        if host==myhostname and drp['active']:
            if supervisor is None:
                # we are supervisor if our pid is the first entry
                supervisor = pid==mypid
            else:
                # only count workers for second and subsequent entries on this host
                nworker+=1
    return supervisor,nworker

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

def opal_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M",verbosity=0):

    global pv
    global cl
    global lm
    global lane
    global xpmpv_global

    print('opal_init')

    lm=lanemask
    lane = (lm&-lm).bit_length()-1
    assert(lm==(1<<lane)) # check that lanemask only has 1 bit for opal
    xpmpv_global = xpmpv
    myargs = { 'dev'         : dev,
               'pollEn'      : False,
               'initRead'    : True,
               'laneConfig'  : {lane:'Opal1000'},
               'dataDebug'   : False,
               'enLclsII'    : True,
               'pgp4'        : False,
               'enableConfig': False,
    }

    # in older versions we didn't have to use the "with" statement
    # but now the register accesses don't seem to work without it -cpo
    cl = cameralink_gateway.ClinkDevRoot(**myargs)
    weakref.finalize(cl, cl.stop)
    cl.start()

    # TODO: To be removed, now commented out xpm glitch workaround
    # Open a new thread here
    #if xpmpv is not None:
    #    cl.ClinkPcie.Hsio.TimingRx.ConfigureXpmMini()
    #    pv = PVCtrls(xpmpv,cl.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper)
    #    pv.start()
    #else:
    #    nbad = 0
    #    while 1:
    #        # check to see if timing is stuck
    #        sof1 = cl.ClinkPcie.Hsio.TimingRx.TimingFrameRx.sofCount.get()
    #        time.sleep(0.1)
    #        sof2 = cl.ClinkPcie.Hsio.TimingRx.TimingFrameRx.sofCount.get()
    #        if sof1!=sof2: break
    #        nbad+=1
    #        print('*** Timing link stuck:',sof1,sof2,'resetting. Iteration:', nbad)
    #        #  Empirically found that we need to cycle to LCLS1 timing
    #        #  to get the timing feedback link to lock
    #        #  cpo: switch this to XpmMini which recovers from more issues?
    #        cl.ClinkPcie.Hsio.TimingRx.ConfigureXpmMini()
    #        time.sleep(3.5)
    #        cl.ClinkPcie.Hsio.TimingRx.ConfigLclsTimingV2()
    #        time.sleep(3.5)

    # camlink timing seems to intermittently lose lock back to the XPM
    # and empirically this fixes it.  not sure if we need the sleep - cpo
    cl.ClinkPcie.Hsio.TimingRx.TimingPhyMonitor.TxPhyReset()
    time.sleep(0.1)

    return cl

def opal_init_feb(slane=None,schan=None):
    # cpo: ignore "slane" because lanemask is given to opal_init() above
    global chan
    if schan is not None:
        chan = int(schan)

# called on alloc
def opal_connectionInfo(cl, alloc_json_str):
    global lane
    global chan

    print('opal_connectionInfo')

    alloc_json = json.loads(alloc_json_str)
    supervisor,nworker = supervisor_info(alloc_json)
    print('camlink supervisor:',supervisor,'nworkers:',nworker)
    barrier_global.init(supervisor,nworker)

    # Open a new thread here
    if xpmpv_global is not None:
        cl.ClinkPcie.Hsio.TimingRx.ConfigureXpmMini()
        pv = PVCtrls(xpmpv_global,cl.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper)
        pv.start()
    else:
        if barrier_global.supervisor:
            nbad = 0
            while 1:
                # check to see if timing is stuck
                sof1 = cl.ClinkPcie.Hsio.TimingRx.TimingFrameRx.sofCount.get()
                time.sleep(0.1)
                sof2 = cl.ClinkPcie.Hsio.TimingRx.TimingFrameRx.sofCount.get()
                if sof1!=sof2: break
                nbad+=1
                print('*** Timing link stuck:',sof1,sof2,'resetting. Iteration:', nbad)
                #  Empirically found that we need to cycle to LCLS1 timing
                #  to get the timing feedback link to lock
                #  cpo: switch this to XpmMini which recovers from more issues?
                cl.ClinkPcie.Hsio.TimingRx.ConfigureXpmMini()
                time.sleep(3.5)
                cl.ClinkPcie.Hsio.TimingRx.ConfigLclsTimingV2()
                time.sleep(3.5)

            # camlink timing seems to intermittently lose lock back to the XPM
            # and empirically this fixes it.  not sure if we need the sleep - cpo
            cl.ClinkPcie.Hsio.TimingRx.TimingPhyMonitor.TxPhyReset()
            time.sleep(0.1)

            txId = timTxId('opal')

            cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
        barrier_global.wait()
        rxId = cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        print('rxId {:x}'.format(rxId))

    if barrier_global.supervisor:
        cl.StopRun()
    barrier_global.wait()
    connect_info = {}
    connect_info['paddr'] = rxId

    # Was opal_connect(cl, connect_json_str):

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

    try:
        connect_info['model'] = opalid.split('-')[1].split('/')[0]
        connect_info['serno'] = opalid.split(':')[1]
    except:
        logging.warning('No opal model/serialnum available on camlink serial port. Configure camera with rogue.')
        connect_info['model'] = 'none'
        connect_info['serno'] = '1000' # not ideal: default to opal1000

    return connect_info

def user_to_expert(cl, cfg, full=False):
    global group

    d = {}
    hasUser = 'user' in cfg
    if (hasUser and 'start_ns' in cfg['user']):
        partitionDelay = getattr(cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
        rawStart       = cfg['user']['start_ns']
        triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
        print('group {:}  partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(group,partitionDelay,rawStart,triggerDelay))
        if triggerDelay < 0:
            print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
            print('Raise start_ns >= {:}'.format(partitionDelay*200*7000/1300))
            raise ValueError('triggerDelay computes to < 0')

        d['expert.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.TriggerDelay']=triggerDelay

    if full:
        d['expert.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.Partition']=group

    if (hasUser and 'gate_ns' in cfg['user']):
        gate = cfg['user']['gate_ns']
        if gate > 160000:
            print('gate_ns {:} may cause errors.  Please use a smaller gate'.format(gate))
            # cpo removed this because people kept asking for it to try
            # to find their signals or increase brightness.  this
            # was originally added because we had an issue where
            # we saw that opal images would intermittently go dark
            # with gates longer than this.
            #raise ValueError('gate_ns > 160000')
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
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path != 'cl' ):
            if 'Hsio.TimingRx' in path and not barrier_global.supervisor:
                print('*** non-supervisor skipping setting',path,'to value',configdb_node)
                continue
            rogue_node.set(configdb_node)
            #  Parameters like black-level need time to take affect (100ms?)

#  Apply the full configuration
def opal_config(cl,connect_str,cfgtype,detname,detsegm,grp):
    global ocfg
    global group
    global lane
    global chan

    print('opal_config')

    group = grp

    appLane  = 'AppLane[%d]'%lane
    clinkFeb = 'ClinkFeb[%d]'%lane
    clinkCh  = 'Ch[%d]'%chan

    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    #if(cl.ClinkPcie.Hsio.PgpMon[lane].RxStatus.RemRxLinkReady.get() != 1): # This is for PGP4
    if(cl.ClinkPcie.Hsio.PgpMon[lane].RxRemLinkReady.get() != 1): # This is for PGP2
        raise ValueError(f'PGP Link is down' )

    applicationLane = getattr(cl.ClinkPcie.Application,appLane)

    # weaver-recommended default for new register for cameralink_gateway 7.11
    if hasattr(applicationLane,'XpmPauseThresh'):
        applicationLane.XpmPauseThresh.set(0xff)

    applicationLane.EventBuilder.Blowoff.set(True)
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

    # should be done by supervisor only, but XpmMini so doesn't really matter
    cl.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.HwEnable.set(False)
    getattr(getattr(cl,clinkFeb).ClinkTop,clinkCh).Blowoff.set(False)
    applicationLane.EventBuilder.Blowoff.set(False)

    # enable all channels in the rogue BatcherEventBuilder
    applicationLane.EventBuilder.Bypass.set(0x0)

    #  Capture the firmware version to persist in the xtc
    cfg['firmwareVersion'] = cl.ClinkPcie.AxiPcieCore.AxiVersion.FpgaVersion.get()
    cfg['firmwareBuild'  ] = cl.ClinkPcie.AxiPcieCore.AxiVersion.BuildStamp.get()

    if barrier_global.supervisor:
        cl.StartRun()

        # supervisor disables all lanes
        # must be done after StartRun because that routine sets MasterEnable
        # to True for all lanes. That causes 100% deadtime from unused lanes.
        for i in range(4):
            cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[i].MasterEnable.set(0)
    barrier_global.wait()
    # enable our lane
    cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[lane].MasterEnable.set(1)

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
    print('opal_unconfig')

    # this routine gets called on disconnect transition
    if barrier_global.supervisor:
        cl.StopRun()
    barrier_global.wait()
    barrier_global.shutdown()

    return cl

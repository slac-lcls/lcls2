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

import pyrogue as pr
import surf.protocols.clink as clink
import rogue.interfaces.stream

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

class MyUartPiranha4Rx(clink.ClinkSerialRx):

    def __init__(self, path):
        super().__init__(path=path)
        self._cur  = []
        self._resp = []

    def _clear(self):
        self._resp = []
        self._cur  = []

    def _check(self):                 # Check if camera is sitting at a prompt
        return ''.join(self._cur) == 'USER>'

    def _await(self, tmo = 5.0):
        tEnd = time.time() + tmo
        while time.time() < tEnd:
            time.sleep(0.01)          # Wait for the Prompt to show up
            if self._check() and len(self._resp):
                return
        raise Exception('*** await: prompt not seen: len %d, resp: %s' % (len(self._resp), self._resp))

    def _acceptFrame(self,frame):
        ba = bytearray(frame.getPayload())
        frame.read(ba,0)

        for i in range(0,len(ba),4):
            c = chr(ba[i])

            if c == '\n':
                #print(self._path+": My Got NL Response: {}".format(''.join(self._cur)))
                self._cur = []
            elif c == '\r':
                last = ''.join(self._cur)
                #print(self._path+": My RecvString: {}".format(last))
                if last != 'USER>':
                    self._resp.append(last)
            elif c != '':
                self._cur.append(c)

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

def piranha4_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M",verbosity=0):

    global pv
    global cl
    global lm
    global lane
    global xpmpv_global

    print('piranha4_init')

    lm=lanemask
    lane = (lm&-lm).bit_length()-1
    assert(lm==(1<<lane)) # check that lanemask only has 1 bit for piranha4
    xpmpv_global = xpmpv
    myargs = { 'dev'         : dev,
               'pollEn'      : False,
               'initRead'    : True,
               'laneConfig'  : {lane:'Piranha4'},
               'dataDebug'   : False,
               'enLclsII'    : True,
               'pgp4'        : False,
               'enableConfig': False,
    }

    # in older versions we didn't have to use the "with" statement
    # but now the register accesses don't seem to work without it -cpo
    cl = cameralink_gateway.ClinkDevRoot(**myargs)

    # Add a custom serial receiver to capture multi-line output
    # Add it here so that it will be used with the inital 'GCP' command
    uart = getattr(getattr(cl,'ClinkFeb[%d]'%lane).ClinkTop,'Ch[%d]'%chan).UartPiranha4
    uart._rx = MyUartPiranha4Rx(uart._rx._path)
    pr.streamConnect(cl.dmaStreams[lane][2],uart._rx)

    # Get ClinkDevRoot.start() and stop() called
    weakref.finalize(cl, cl.stop)
    cl.start()

    # TODO: To be removed, now commented out xpm glitch workaround
    ## Open a new thread here
    #if xpmpv is not None:
    #    cl.ClinkPcie.Hsio.TimingRx.ConfigureXpmMini()
    #    pv = PVCtrls(xpmpv,cl.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper)
    #    pv.start()
    #else:
    #    #  Empirically found that we need to cycle to LCLS1 timing
    #    #  to get the timing feedback link to lock
    #    #  cpo: switch this to XpmMini which recovers from more issues?
    #    # check to see if timing is stuck
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

def piranha4_init_feb(slane=None,schan=None):
    # cpo: ignore "slane" because lanemask is given to piranha4_init() above
    global chan
    if schan is not None:
        chan = int(schan)

# called on alloc
def piranha4_connectionInfo(cl, alloc_json_str):
    global lane
    global chan

    print('piranha4_connectionInfo')

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

            txId = timTxId('piranha4')

            cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
        barrier_global.wait()
        rxId = cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
        print('rxId {:x}'.format(rxId))

    if barrier_global.supervisor:
        cl.StopRun()
    barrier_global.wait()
    connect_info = {}
    connect_info['paddr'] = rxId

    # Was piranha4_connect(cl, connect_json_str):

    ## initialize the serial link
    #uart = getattr(getattr(cl,'ClinkFeb[%d]'%lane).ClinkTop,'Ch[%d]'%chan)
    #uart.BaudRate.set(9600)
    #uart.SerThrottle.set(10000)
    #time.sleep(0.10)

    # Startup's GCP returns 'Model  P4_CM_0xKxxD_00_R' and 'Serial #  xxxxxxxx', etc.
    uart = getattr(getattr(cl,'ClinkFeb[%d]'%lane).ClinkTop,'Ch[%d]'%chan).UartPiranha4

    uart._rx._await()           # Wait for camera to be sitting at a prompt

    uart._rx._clear()
    uart.SEM.set('0') # Set internal exposure mode for quicker commanding (?!)
    uart._rx._await()
    uart._rx._clear()
    uart.GCP()
    uart._rx._await()

    t0 = time.time()
    while len(uart._rx._resp) == 0 or not uart._rx._resp[-1].startswith('CPA'):
        time.sleep(.01)
        if time.time() - t0 > 5.0:
            print("Last response:")
            print(uart._rx._resp)
            raise Exception("Response to 'GCP' not seen")

    model = ''
    serno = ''
    bist  = ''
    for line in uart._rx._resp:
        if   line.startswith('Model'):     model = line.split()[-1]
        elif line.startswith('Serial #'):  serno = line.split()[-1]
        elif line.startswith('BiST'):      bist  = line.split()[-1]

    print('model {:}'.format(model))
    print('serno {:}'.format(serno))
    print('bist {:}' .format(bist))
    if len(bist) and bist != 'Good':
        print('Piranha BiST error: Check User\'s manual for meaning')

    try:
        connect_info['model'] = model if model == '' else (model.split('_')[2].split('K')[0])
        connect_info['serno'] = serno
        connect_info['bist']  = bist
    except:
        logging.warning('No piranha model/serialnum available on camlink serial port. Configure camera with rogue.')
        connect_info['model'] = 'none'
        connect_info['serno'] = 'P4_CM_02K10D_00_R' # not ideal: default to P4_CM_02K10D_00_R
        connect_info['bist']  = 'Bad'

    uart._rx._clear()
    uart.VT()
    uart._rx._await()
    print('Temperature: ', uart._rx._resp[-1])

    uart._rx._clear()
    uart.VV()
    uart._rx._await()
    print('Voltage: ', uart._rx._resp[-1])

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
        if gate < 4000:
            print('gate_ns {:} must be at least 4000 ns'.format(gate))
            raise ValueError('gate_ns < 4000')
        if gate > 160000:
            print('gate_ns {:} may cause errors.  Please use a smaller gate'.format(gate))
            #raise ValueError('gate_ns > 160000')
        d['expert.ClinkFeb.TrigCtrl.TrigPulseWidth']=1.0 #gate*0.001
        d['expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SET']=gate

    if (hasUser and 'black_level' in cfg['user']):
        d['expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SSB']=cfg['user']['black_level']

    if (hasUser and 'vertical_bin' in cfg['user']):
        d['expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SBV']=cfg['user']['vertical_bin']

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
    uart = getattr(getattr(cl,'ClinkFeb[%d]'%lane).ClinkTop,'Ch[%d]'%chan).UartPiranha4
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
            if 'UartPiranha4' in str(rogue_node):
                uart._rx._clear()
            if 'Hsio.TimingRx' in path and not barrier_global.supervisor:
                print('*** non-supervisor skipping setting',path,'to value',configdb_node)
                continue
            rogue_node.set(configdb_node)
            #  Parameters like black-level need time to take affect (up to 1.75s)
            if 'UartPiranha4' in str(rogue_node):
                uart._rx._await()

#  Apply the full configuration
def piranha4_config(cl,connect_str,cfgtype,detname,detsegm,grp):
    global ocfg
    global group
    global lane
    global chan

    print('piranha4_config')

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

    # drain any data in the event pipeline
    applicationLane.EventBuilder.Blowoff.set(True)
    getattr(getattr(cl,clinkFeb).ClinkTop,clinkCh).Blowoff.set(True)

    #  set bool parameters
    cfg['expert']['ClinkFeb']['TrigCtrl']['EnableTrig'] = True
    cfg['expert']['ClinkFeb']['TrigCtrl']['InvCC'] = False
    cfg['expert']['ClinkFeb']['ClinkTop']['ClinkCh']['DataEn'] = True

    user_to_expert(cl,cfg,full=True)

    config_expert(cl,cfg['expert'])

    uart = getattr(getattr(cl,clinkFeb).ClinkTop,clinkCh).UartPiranha4

    uart._rx._clear()
    uart.VT()
    uart._rx._await()
    #print('Temperature: ', uart._rx._resp[-1])

    uart._rx._clear()
    uart.VV()
    uart._rx._await()
    #print('Voltage: ', uart._rx._resp[-1])

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
        # Empirically found that StartRun must be done before externally triggered mode
        # is enabled or the Piranha goes into an error state and causes deadtime.
        # The sequence 'sem 0', 'set 4000', 'stm 0', stm 1' clears the error state so
        # the configDb sequence (piranha4_config_store.py) and this code is arranged
        # to reproduce that.
        cl.StartRun()

        # supervisor disables all lanes
        # must be done after StartRun because that routine sets MasterEnable
        # to True for all lanes. That causes 100% deadtime from unused lanes.
        for i in range(4):
            cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[i].MasterEnable.set(0)
    barrier_global.wait()

    uart._rx._clear()
    uart.STM.set('1')  # set to externally triggered mode
    uart._rx._await()

    ## We want to use external exposure mode so that the exposure is driven by
    ## the length of the CC1 pulse.  Unfortunately this setting makes commanding
    ## the Piranha much slower (2+ sec per command), so we issue this one last.
    #uart._rx._clear()
    #uart.SEM.set('1')  # set to exposure mode to external
    #uart._rx._await()

    # enable our lane
    cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[lane].MasterEnable.set(1)

    ocfg = cfg
    return json.dumps(cfg)

    #ncfg = cfg.copy()
    #del ncfg['expert']['ClinkFeb']
    #return json.dumps(ncfg)


def piranha4_scan_keys(update):
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

def piranha4_update(update):
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

def piranha4_unconfig(cl):
    print('piranha4_unconfig')

    # this routine gets called on disconnect transition
    if barrier_global.supervisor:
        cl.StopRun()
    barrier_global.wait()
    barrier_global.shutdown()

    return cl

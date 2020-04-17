from psalg.configdb.get_config import get_config
from .xpmmini import *
import rogue
import cameralink_gateway
import socket
import time
import json
import IPython
from collections import deque

#this function reads the data base and converts it into.... what sort of object?
def read_database():
    return 0

#this takes the data base object from read_database, does some messy calculations on it, and converts it to a rogue writeable object
def database_object_calculations_2rogue():
    return 0


#this function takes in the object from
def write_to_rogue():
    return 0

pv = None

def opal_init(xpmpv=None):

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
        pv = PVCtrls(xpmpv,cl.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper)
        pv.start()

    return cl

def opal_connect(cl):

    ip = socket.inet_aton(socket.gethostbyname(socket.gethostname()))
    txId = (0xf9<<24) | (ip[2]<<8) | (ip[3])

    lane = 0
    chan = 0

    rxId = cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)

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

    # overwrite the low-level configuration parameters with calculations from the user configuration

    partitionDelay = getattr(cl.ClinkPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
    rawStart       = cfg['user']['start_ns']
    triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
    print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
    if triggerDelay < 0:
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        raise ValueError('triggerDelay computes to < 0')

    cfg['expert']['ClinkPcie']['Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer[0]']['TriggerDelay'] = triggerDelay

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

    #  Capture the firmware version to persist in the xtc
    cfg['firmwareVersion'] = cl.ClinkPcie.AxiPcieCore.AxiVersion.FpgaVersion.get()
    cfg['firmwareBuild'  ] = cl.ClinkPcie.AxiPcieCore.AxiVersion.BuildStamp.get()

    cl.StartRun()

    return json.dumps(cfg)

if __name__ == "__main__":


    print(20*'_')
    print(20*'_')
    print("Executing main")
    print(20*'_')
    print(20*'_')

    connect_info = {}
    connect_info['body'] = {}
    connect_info['body']['control'] = {}
    connect_info['body']['control']['0'] = {}
    connect_info['body']['control']['0']['control_info'] = {}
    connect_info['body']['control']['0']['control_info']['instrument'] = 'TST'
    connect_info['body']['control']['0']['control_info']['cfg_dbase'] = 'mcbrowne:psana@psdb-dev:9306/sioanDB'

    mystring = json.dumps(connect_info)                             #paste this string into the pgpread_timetool.cc as parameter for the opal_config function call
    print(mystring)
    print(20*'_')
    print(20*'_')
    print("Calling opal_config")
    print(20*'_')
    print(20*'_')

    my_config = opal_config(mystring,"BEAM", "tmotimetool")

    print(20*'_')
    print(20*'_')
    print("opal_config finished")
    print(20*'_')
    print(20*'_')

    print(my_config)

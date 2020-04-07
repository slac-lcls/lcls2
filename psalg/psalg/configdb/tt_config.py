from psalg.configdb.get_config import get_config
import rogue
import lcls2_timetool
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

def tt_config(connect_str,cfgtype,detname,detsegm,group):
    cfg = get_config(connect_str,cfgtype,detname,detsegm)

    myargs = { 'dev'         : '/dev/datadev_0',
               'pgp3'        : False,
               'pollEn'      : False,
               'initRead'    : True,
               'dataCapture' : False,
               'dataDebug'   : False,}

    # in older versions we didn't have to use the "with" statement
    # but now the register accesses don't seem to work without it -cpo
    with lcls2_timetool.TimeToolKcu1500Root(**myargs) as cl:

        if(cl.TimeToolKcu1500.Kcu1500Hsio.PgpMon[0].RxRemLinkReady.get() != 1):
            raise ValueError(f'PGP Link is down' )

        # TimeToolKcu1500Root.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId
        partitionDelay = getattr(cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,'PartitionDelay[%d]'%group).get()
        rawStart       = cfg['user']['start_ns']
        triggerDelay   = int(rawStart*1300/7000 - partitionDelay*200)
        print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
        if triggerDelay < 0:
            print('partitionDelay {:}  rawStart {:}  triggerDelay {:}'.format(partitionDelay,rawStart,triggerDelay))
            raise ValueError('triggerDelay computes to < 0')
            
        cfg['cl']['TimeToolKcu1500']['Kcu1500Hsio']['TimingRx']['TriggerEventManager']['TriggerEventBuffer0']['TriggerDelay'] = triggerDelay
        
        cl.ClinkFeb[0].ClinkTop.Ch[0].UartPiranha4.SendEscape()

        # traverse daq config database tree and print corresponding
        # rogue value
        # doing this means that fields can't manually be added to the
        # daq config unless the person doing so knows what the axi
        # lite registers are

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

                if("UartPiranha4" in path):
                    #UartPiranhaCode  = (path.split(".")[-1]).lower()
                    #UartValue = cl.ClinkFeb[0].ClinkTop.Ch[0].UartPiranha4._tx.sendString("get '"+UartPiranhaCode)
                    #print(path+", rogue value = "+str(UartValue)+", daq config database = " +str(configdb_node))
                    rogue_node.set(int(str(configdb_node),10))
                    pass

                else:
                    print(path+", rogue value = "+str(hex(rogue_node.get()))+", daq config database = " +str(configdb_node))

                    # this is where the magic happens.  I.e. this is where the rogue axi lite register is set to the daq config database value
                    # There's something uneasy about this
                    rogue_node.set(int(str(configdb_node),16))
    
        cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.XpmMiniWrapper.XpmMini.HwEnable.set(True)
        cl.TimeToolKcu1500.Kcu1500Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable.set(True)

        #  Capture the firmware version to persist in the xtc
        cfg['firmwareVersion'] = cl.TimeToolKcu1500.AxiPcieCore.AxiVersion.FpgaVersion.get()
        cfg['firmwareBuild'  ] = cl.TimeToolKcu1500.AxiPcieCore.AxiVersion.BuildStamp.get()

        #cl.StartRun()

        #cl.stop()   #gui.py should be able to run after this line, but it's still using the axi lite resource.
        #deleting cl doesn't resolve this problem.

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

    mystring = json.dumps(connect_info)                             #paste this string into the pgpread_timetool.cc as parameter for the tt_config function call
    print(mystring)
    print(20*'_')
    print(20*'_')
    print("Calling tt_config")
    print(20*'_')
    print(20*'_')

    my_config = tt_config(mystring,"BEAM", "tmotimetool")

    print(20*'_')
    print(20*'_')
    print("tt_config finished")
    print(20*'_')
    print(20*'_')

    print(my_config)

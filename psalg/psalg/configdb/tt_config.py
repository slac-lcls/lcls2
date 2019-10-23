from psalg.configdb.get_config import get_config
import rogue
import TimeToolDev
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

def tt_config(connect_str,cfgtype,detname,group):

    cfg = get_config(connect_str,cfgtype,detname)


    #toggle_prescaling()


    #################################################################
    try:
        cl = TimeToolDev.TimeToolDev(
            dev       = '/dev/datadev_0',
            dataDebug = False,
            version3  = False,
            pollEn    = False,
            initRead  = False,
            enVcMask  = 0xD,
        )
    except rogue.GeneralError:
        #print("rogue.GeneralError: AxiStreamDma::AxiStreamDma: General Error: failed to open file /dev/datadev_0 with dest 0x0 terminate called after throwing an instance of 'char const*'")
        print("ERROR: Close any other applications using rogue to communicate with AXI Lite registers ")
        raise
        

    #################################################################

    if(cl.Hardware.PgpMon[0].RxRemLinkReady.get() != 1):
        raise ValueError(f'PGP Link is down' )
        
    #################################################################


    ###############################################################################
    ### traverse daq config database tree and print corresponding rogue value #####
    ###############################################################################
    # doing this means that fields can't manually be added to the daq config unless the person doing so knows what the axi lite registers are

    depth = 0
    path  = 'cl'
    my_queue  =  deque([[path,depth,cl,cfg['cl']]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        if(dict is type(configdb_node)):
            for i in configdb_node:
                my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[i],configdb_node[i]])
        
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path is not 'cl' ):
            #print(path)
            print(path+", rogue value = "+str(hex(rogue_node.get()))+", daq config database = " +str(configdb_node))
    


    ##############
    #####
    ##############

    
    scratch_pad = (cfg['cl']['Application']['AppLane[0]']['Prescale']['ScratchPad'])   

    cl.Application.AppLane[0].Prescale.ScratchPad.set(scratch_pad)                       #writing to rogue register 

    print("scratch pad value = ",cl.Application.AppLane[0].Prescale.ScratchPad.get())

    cl.stop()   #gui.py should be able to run after this, but it's still using the axi lite resource.
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
    connect_info['body']['control']['0']['control_info']['instrument'] = 'TMO'
    connect_info['body']['control']['0']['control_info']['cfg_dbase'] = 'mcbrowne:psana@psdb-dev:9306/sioanDB'

    mystring = json.dumps(connect_info)                             #paste this string into the pgpread_timetool.cc as parameter for the tt_config function call
    print(mystring)
    print(20*'_')
    print(20*'_')
    print("Calling tt_config")
    print(20*'_')
    print(20*'_')

    my_config = tt_config(mystring,"BEAM", "tmotimetool",None)          

    print(20*'_')
    print(20*'_')
    print("tt_config finished")
    print(20*'_')
    print(20*'_')

    print(my_config)

"""
(lcls2daq_ttdep) [sioan@lcls-pc83236 lcls2]$ python psalg/psalg/configdb/tt_config.py 
Traceback (most recent call last):
  File "psalg/psalg/configdb/tt_config.py", line 16, in <module>
    my_string = tt_config("mcbrowne:psana@psdb-dev:9306",'BEAM', 'tmotimetool',None)
  File "psalg/psalg/configdb/tt_config.py", line 7, in tt_config
    cfg = get_config(connect_str,cfgtype,detname)
  File "/u1/sioan/miniconda3/envs/lcls2daq_ttdep/lib/python3.7/site-packages/psalg/configdb/get_config.py", line 25, in get_config
    connect_info = json.loads(connect_json)
  File "/u1/sioan/miniconda3/envs/lcls2daq_ttdep/lib/python3.7/json/__init__.py", line 348, in loads
    return _default_decoder.decode(s)
  File "/u1/sioan/miniconda3/envs/lcls2daq_ttdep/lib/python3.7/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/u1/sioan/miniconda3/envs/lcls2daq_ttdep/lib/python3.7/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
"""

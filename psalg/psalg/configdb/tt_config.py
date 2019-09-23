from psalg.configdb.get_config import get_config
import TimeToolDev
import json


def tt_config(connect_str,cfgtype,detname,group):

    cfg = get_config(connect_str,cfgtype,detname)


    #toggle_prescaling()


    #################################################################
    cl = TimeToolDev.TimeToolDev(
        dev       = '/dev/datadev_0',
        dataDebug = False,
        version3  = False,
        pollEn    = False,
        initRead  = False,
        enVcMask  = 0xD,
    )

    #################################################################

    if(cl.Hardware.PgpMon[0].RxRemLinkReady.get() != 1):
        raise ValueError(f'PGP Link is down' )
        
    #################################################################
    
    scratch_pad = (cfg['cl']['Application']['AppLane1']['Prescale']['ScratchPad'])

    cl.Application.AppLane[0].Prescale.ScratchPad.set(scratch_pad)

    print("scratch pad value = ",cl.Application.AppLane[0].Prescale.ScratchPad.get())

    cl.stop()

    return json.dumps(cfg)

if __name__ == "__main__":


    connect_info = {}
    connect_info['body'] = {}
    connect_info['body']['control'] = {}
    connect_info['body']['control']['0'] = {}
    connect_info['body']['control']['0']['control_info'] = {}
    connect_info['body']['control']['0']['control_info']['instrument'] = 'TMO'
    connect_info['body']['control']['0']['control_info']['cfg_dbase'] = 'mcbrowne:psana@psdb-dev:9306/configDB'
    import json
    mystring = json.dumps(connect_info)                             #paste this string into the pgpread_timetool.cc as parameter for the tt_config function call
    print(mystring)
    my_config = tt_config(mystring,"BEAM", "tmotimetool",None)          
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

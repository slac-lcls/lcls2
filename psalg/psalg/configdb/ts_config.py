from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
import json
import time

def ts_config(connect_json,pv_postfix,cfgtype,detname):

    cfg = get_config(connect_json,cfgtype,detname)
    connect_info = json.loads(connect_json)
    print('*** ts_config cfg',cfg)

    # FIXME: need to use the pv_prefix to configure things like trig rate
    control_info = connect_info['body']['control']['0']['control_info']
    pv_prefix = control_info['pv_base']+':'+pv_postfix
    print('*** ts_config pv_prefix:',pv_prefix)

    return json.dumps(cfg)

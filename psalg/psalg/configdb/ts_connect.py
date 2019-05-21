from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
import json
import time

class ts_connector:
    def __init__(self,json_connect_info):
        self.connect_info = json.loads(json_connect_info)
        print('*** connect_info',self.connect_info)

        # get the base from the collection?
        self.xpm_base = 'DAQ:LAB2:XPM:'

        self.ctxt = Context('pva')
        self.get_xpm_ports()
        self.get_readout_group_mask()

        # unfortunately, the hsd needs the Rx link reset before the Tx,
        # otherwise we get CRC errors on the link.
        self.xpm_link_reset('Rx')
        self.xpm_link_reset('Tx')

        # must come after the link reset because it uses the links
        self.clear_readout()
    
        # must come after clear readout because clear readout increments
        # the event counters, and the pgp eb needs them to start from zero
        self.l0_count_reset()

        self.ctxt.close()

    def get_readout_group_mask(self):
        self.readout_group_mask = 0
        for key,node_info in self.connect_info['body']['drp'].items():
            try:
                self.readout_group_mask |= node_info['det_info']['readout']
            except KeyError:
                pass

    def get_xpm_ports(self):
        self.xpm_ports = []
        for key,node_info in self.connect_info['body']['drp'].items():
            try:
                xpm_id = int(node_info['connect_info']['xpm_ip'].split('.')[2])
                xpm_port = node_info['connect_info']['xpm_port']
                self.xpm_ports.append((xpm_id,xpm_port))
            except KeyError:
                pass

    def xpm_link_reset(self,style):
        names = []
        # make pv name that looks like DAQ:LAB2:XPM:1:RxLinkReset11
        # for xpm_num 1 and xpm_port 11
        for xpm_num,xpm_port in self.xpm_ports:
            pvname = self.xpm_base+str(xpm_num)+':'+style+'LinkReset'+str(xpm_port)
            names.append(pvname)
        self.ctxt.put(names,len(names)*[1])
        # unfortunately need to wait for the links to relock, which
        # matt says takes "an appreciable fraction of a second"
        time.sleep(1)

    def clear_readout(self):
        # sending TransitionId 0 does clear readout
        pass

    def l0_count_reset(self):
        l0reset_name = self.xpm_base+':'+'GroupL0Reset'
        pass

def ts_connect(json_connect_info):

    connector = ts_connector(json_connect_info)
    return json.dumps({})

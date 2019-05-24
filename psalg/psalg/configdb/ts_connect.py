from psalg.configdb.get_config import get_config
from p4p.client.thread import Context
from psdaq.control.collection import DaqControl

import json
import time

class ts_connector:
    def __init__(self,json_connect_info):
        self.connect_info = json.loads(json_connect_info)
        print('*** connect_info',self.connect_info)

        control_info=self.connect_info['body']['control']['0']['control_info']
        self.xpm_base = control_info['pv_base']+':XPM:'
        master_xpm_num = control_info['xpm_master']
        self.master_xpm_pv = self.xpm_base+str(master_xpm_num)+':'

        self.ctxt = Context('pva')
        self.get_xpm_info()
        self.get_readout_group_mask()

        # unfortunately, the hsd needs the Rx link reset before the Tx,
        # otherwise we get CRC errors on the link.
        #self.xpm_link_reset('Rx')
        #self.xpm_link_reset('Tx')

        # must come after clear readout because clear readout increments
        # the event counters, and the pgp eb needs them to start from zero
        self.l0_count_reset()

        # at the moment, clearing and setting the link enables messes
        # up the link, so commenting out for now.
        # enables listening to deadtime
        # self.xpm_link_enable()

        self.ctxt.close()

    def get_readout_group_mask(self):
        self.readout_group_mask = 0
        for _,_,readout_group in self.xpm_info:
            self.readout_group_mask |= (1<<readout_group)

    def get_xpm_info(self):
        self.xpm_info = []
        for key,node_info in self.connect_info['body']['drp'].items():
            try:
                # FIXME: should have a better method to map xpm ip
                # address to xpm number (used to create pv names)
                xpm_id = int(node_info['connect_info']['xpm_ip'].split('.')[2])
                xpm_port = node_info['connect_info']['xpm_port']
                readout_group = node_info['det_info']['readout']
                self.xpm_info.append((xpm_id,xpm_port,readout_group))
            except KeyError:
                pass

    def xpm_link_disable_all(self):
        # FIXME: need a mechanism to disable unused links in all
        # downstream XPMs. For now, just clear out our readout
        # groups from all the XPMs we know about from the collection,
        # which comes from the "remote link id" info in the drp nodes.
        xpms = [xpm_num for xpm_num,_,_ in self.xpm_info]
        unique_xpms = set(xpms)
        pv_names = []
        for xpm_num in unique_xpms:
            for xpm_port in range(32):
                pv_names.append(self.xpm_base+str(xpm_num)+':'+'LinkGroupMask'+str(xpm_port))
        current_group_masks = self.ctxt.get(pv_names)

        print(current_group_masks)
        # don't clear out group_mask 0xff (an indication that it's
        # a downstream XPM link)
        pv_names_to_clear = [pv_name for (pv_name,group_mask) in zip(pv_names,current_group_masks) if (group_mask & self.readout_group_mask) and (group_mask != 0xff)]
        print('*** clearing xpm links',pv_names_to_clear)
        self.ctxt.put(pv_names_to_clear,len(pv_names_to_clear)*[0])

    def xpm_link_enable(self):
        self.xpm_link_disable_all()

        pv_names = []
        values = []
        for xpm_num,xpm_port,readout_group in self.xpm_info:
            pvname = self.xpm_base+str(xpm_num)+':'+'LinkGroupMask'+str(xpm_port)
            pv_names.append(pvname)
            values.append((1<<readout_group))
        print('*** setting xpm link enables',pv_names,values)
        self.ctxt.put(pv_names,values)

    def xpm_link_reset(self,style):
        # make pv name that looks like DAQ:LAB2:XPM:1:RxLinkReset11
        # for xpm_num 1 and xpm_port 11
        pv_names = []
        for xpm_num,xpm_port,_ in self.xpm_info:
            pvname = self.xpm_base+str(xpm_num)+':'+style+'LinkReset'+str(xpm_port)
            pv_names.append(pvname)
        print('*** xpm link resetting',pv_names)
        self.ctxt.put(pv_names,len(pv_names)*[1])
        # unfortunately need to wait for the links to relock, which
        # matt says takes "an appreciable fraction of a second".
        # empirically, the links seem unreliable unless we wait 2s.
        time.sleep(2)

    def l0_count_reset(self):
        pvL0Reset = self.master_xpm_pv+'GroupL0Reset'
        print('*** resetting l0 count',self.readout_group_mask)
        self.ctxt.put(pvL0Reset,self.readout_group_mask)


def ts_connect(json_connect_info):

    connector = ts_connector(json_connect_info)
    return json.dumps({})

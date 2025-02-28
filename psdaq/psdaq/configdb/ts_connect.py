from psdaq.configdb.get_config import get_config
from p4p.client.thread import Context

import json
import time
import pprint

class xpm_link:
    def __init__(self,value):
        self.value = value

    def is_xpm(self):
        return (int(self.value)>>24)&0xff == 0xff

    def xpm_num(self):
        print('xpm_num {:x} {:}'.format(self.value,(int(self.value)>>16)&0xff))
        return (int(self.value)>>16)&0xff

class ts_connector:
    def __init__(self,json_connect_info):
        self.connect_info = json.loads(json_connect_info)
        print('*** connect_info')
        pp = pprint.PrettyPrinter()
        pp.pprint(self.connect_info)

        control_info=self.connect_info['body']['control']['0']['control_info']
        self.xpm_base = control_info['pv_base']+':XPM:'
        master_xpm_num = control_info['xpm_master']
        self.master_xpm_pv = self.xpm_base+str(master_xpm_num)+':'

        self.ctxt = Context('pva')
        self.get_xpm_info()
        self.get_readout_group_mask()

        # unfortunately, the hsd needs the Rx link reset before the Tx,
        # otherwise we get CRC errors on the link.
        # try commenting this out since Matt has made the links more reliable
        #self.xpm_link_reset('Rx')
        #self.xpm_link_reset('Tx')

        # must come after clear readout because clear readout increments
        # the event counters, and the pgp eb needs them to start from zero
        # comment this out since it was moved to control.py
        #self.l0_count_reset()

        # enables listening to deadtime
        self.xpm_link_enable()

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
                xpm_id = int(node_info['connect_info']['xpm_id'])
                xpm_port = node_info['connect_info']['xpm_port']
                readout_group = node_info['det_info']['readout']
                self.xpm_info.append((xpm_id,xpm_port,readout_group))
            except KeyError:
                pass

    def xpm_link_disable(self, pv, groups):
        pv_names = []
        for xpm_port in range(14):
            pv_names.append(pv+'RemoteLinkId' +str(xpm_port))
        print('link_ids: {:}'.format(pv_names))
        try:
            link_ids = self.ctxt.get(pv_names)
        except TimeoutError:
            print('*** TimeoutError for {:}'.format(pv))
            return

        pv_names = []
        downstream_xpm_names = []
        for xpm_port in range(14):
            pv_names.append(pv+'LinkGroupMask'+str(xpm_port))
        link_masks = self.ctxt.get(pv_names)

        for i in range(14):
            xlink = xpm_link(link_ids[i])
            # this gets run for all xpm's "downstream" of the master xpm
            if xlink.is_xpm():
                downstream_xpm_names.append(self.xpm_base+str(xlink.xpm_num()))
                self.xpm_link_disable(self.xpm_base+str(xlink.xpm_num())+':',groups)
                link_masks[i] = 0xff   # xpm to xpm links should be enabled for everything
            else:
                link_masks[i] &= ~groups

        self.ctxt.put(pv_names,link_masks)

        # this code disables the "master" feature for each of the
        # downstream xpm's for the readout groups used by the new xpm master
        pv_names_downstream_xpm_master_enable = []
        for name in downstream_xpm_names:
            for igroup in range(8):
                if (1<<igroup)&groups:
                    pv_names_downstream_xpm_master_enable.append(name+':PART:%d:Master'%igroup)
            # also need to disable the sequences running on the downstream xpms 
            for iseq in range(8):
                pv_names_downstream_xpm_master_enable.append(name+':SEQENG:%d:ENABLE'%iseq)
       
        num_master_disable = len(pv_names_downstream_xpm_master_enable)
        if (num_master_disable):
            print('*** Disable downstream xpm readout group master+sequences:',pv_names_downstream_xpm_master_enable)
            try:
                self.ctxt.put(pv_names_downstream_xpm_master_enable,[0]*num_master_disable)
            except TimeoutError:
                print('*** TimeoutError')

    def xpm_link_disable_all(self):
        # Start from the master and recursively remove the groups from each downstream link
        self.xpm_link_disable(self.master_xpm_pv, self.readout_group_mask)

    def xpm_link_enable(self):
        self.xpm_link_disable_all()

        d = {}
        pvnames_rxready = []
        for xpm_num,xpm_port,readout_group in self.xpm_info:
            pvname = self.xpm_base+str(xpm_num)+':'+'LinkGroupMask'+str(xpm_port)
            pvnames_rxready.append(self.xpm_base+str(xpm_num)+':'+'LinkRxReady'+str(xpm_port))
            if pvname in d:
                d[pvname] |= (1<<readout_group)
            else:
                d[pvname] = (1<<readout_group)

        # make sure the XPM links from the detectors are OK
        cnt = 0
        while cnt < 15:
            cnt += 1
            rxready_values = self.ctxt.get(pvnames_rxready)
            for pvname,val in zip(pvnames_rxready,rxready_values):
                if val==1:  break
                time.sleep(1)
            if val==1:  break
        if val == 1:
            if cnt > 1:  print('RxLink locked after %u tries' % cnt)
        else:
            raise ValueError('RxLink not locked! %s' % pvname)

        pv_names = []
        values = []
        for name,value in d.items():
            pv_names.append(name)
            values.append(value)

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

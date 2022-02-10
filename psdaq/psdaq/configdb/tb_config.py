from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.cas.xpm_utils import timTxId
from p4p.client.thread import Context
import sys
import json
import time
import rogue
import logging

#
#  I need to import some project to get the surf library
#  This library may be a different version in other projects
#
import lcls2_pgp_pcie_apps

#
#  Still need to configure lcls2_pgp_pcie_apps BEB for our lane and no PGP
#

base = None
ocfg = None
pv_prefix = None
readout_groups = None
lane = 3

import pyrogue as pr
import lcls2_pgp_fw_lib.shared as shared
import axipcie
import surf.axi as axi
import surf.devices.transceivers as xceiver

class _Device(pr.Device):
    def __init__(self,
                 description = 'Just AxiVersion',
                 **kwargs):
        super().__init__(description=description, **kwargs)

        self.add(axipcie.PcieAxiVersion(
            offset = 0x20000,
            expand = False,
        ))

        self.add(axi.AxiLiteMasterProxy(
            name   = 'AxilBridge',
            offset = 0x70000,
        ))

        qsfpOffset = [0x74_000,0x71_000]

        for i in range(2):
            self.add(xceiver.Qsfp(
                name    = f'Qsfp[{i}]',
                offset  = qsfpOffset[i],
                memBase = self.AxilBridge.proxy,
                enabled = False, # enabled=False because I2C are slow transactions and might "log jam" register transaction pipeline
            ))


#
#  Create a generic DevRoot to query the firmware type
#
class _DevRoot(shared.Root):
    def __init__(self,
                 dev     = '/dev/datadev_0',
                 **kwargs):

        super().__init__(
            dev      = dev,
            pollEn   = False,
            initRead = False,
            **kwargs)

        self.memMap = axipcie.createAxiPcieMemMap(dev, 'localhost', 8000)
        self.memMap.setName('PCIe_Bar0')
        self.add(_Device(name       = 'DevPcie',
                         memBase    = self.memMap))

    def start(self, **kwargs):
        super().start(**kwargs)

def tb_init(arg,dev='/dev/datadev_0',lanemask=1,xpmpv=None,timebase="186M",verbosity=0):
    global base
    logging.debug('tb_init')

    #
    #  Invoke generic root to lookup firmware type
    #
    with _DevRoot(dev = dev) as _base:
        fwImage = _base.DevPcie.AxiVersion.ImageName.get()
    #  Some images have conflicting versions of surf module
    sys.module.pop('surf')

    print(f'Found image {fwImage}')
 
    base = {}
    base['fwImage'] = fwImage
    base['virtChan'] = 1

    if fwImage.startswith('Lcls2Xilinx'):
        import lcls2_pgp_pcie_apps
        pbase = lcls2_pgp_pcie_apps.DevRoot(dev           =dev,
                                            enLclsI       =False,
                                            enLclsII      =True,
                                            yamlFileLclsI =None,
                                            yamlFileLclsII=None,
                                            startupMode   =True,
                                            standAloneMode=False,
                                            pgp4          =True,
                                            dataVc        =0,
                                            pollEn        =False,
                                            initRead      =False)
        base['virtChan'] = 0
    elif fwImage.startswith('Lcls2EpixHr'):
        import lcls2_epix_hr_pcie
        pbase = lcls2_epix_hr_pcie.DevRoot(dev           =dev,
                                           enLclsI       =False,
                                           enLclsII      =True,
                                           yamlFileLclsI =None,
                                           yamlFileLclsII=None,
                                           startupMode   =True,
                                           standAloneMode=False,
                                           pgp4          =True,
                                           #dataVc        =0,
                                           pollEn        =False,
                                           initRead      =False)
    elif fwImage.startswith('Clink'):
        import cameralink_gateway
        pbase = cameralink_gateway.ClinkDevRoot(dev         =dev,
                                                pollEn      =False,
                                                initRead    =True,
                                                laneConfig  ={0:'Opal1000'},
                                                dataDebug   =False,
                                                enLclsII    =True,
                                                pgp4        =False,
                                                enableConfig=False)
    else:
        raise ValueError(f'Image not recognized - {fwImage}')

    pbase.__enter__()

    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ModeSelEn.set(1)
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ModeSel.set(1)
    if timebase=="119M":
        logging.info('Using timebase 119M')
        base['clk_period'] = 1000/119. 
        base['msg_period'] = 238
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(0)
    else:
        logging.info('Using timebase 186M')
        base['clk_period'] = 7000/1300. # default 185.7 MHz clock
        base['msg_period'] = 200
        pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.ClkSel.set(1)
    pbase.DevPcie.Hsio.TimingRx.TimingFrameRx.RxDown.set(0)
    base['pcie'] = pbase
    return base

def tb_init_feb(slane=None,schan=None):
    global lane
    global chan
    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)
    print(f'init_feb lane {lane}')

def tb_connect(base):
    pbase = base['pcie']
    rxId = pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.RxId.get()
    logging.info('RxId {:x}'.format(rxId))
    txId = timTxId('tdet')
    logging.info('TxId {:x}'.format(txId))
    pbase.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner.TxId.set(txId)
    fwImage = base['fwImage']
    if (fwImage.startswith('Lcls2Xilinx') or
        fwImage.startswith('Clink')):
        getattr(pbase.DevPcie.Application,f'AppLane[{lane}]').EventBuilder.Bypass.set(0x2)
    elif fwImage.startswith('Lcls2EpixHr'):
        pbase.DevPcie.Application.EventBuilder.Bypass.set(0x3c)
    else:
        raise ValueError(f'Image not recognized - {fwImage}')

    d = {}
    d['paddr'] = rxId
    d['serno'] = '-'
    return d


def tb_config(base,connect_str,cfgtype,detname,detsegm,rog):
    global ocfg
    global pv_prefix
    global readout_groups

    print('tb_config')
    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    ocfg = cfg

    # get the list of readout groups that the user has selected
    # so we only configure those
    readout_groups = []
    connect_info = json.loads(connect_str)
    for nodes in connect_info['body']['drp'].values():
        readout_groups.append(nodes['det_info']['readout'])
    readout_groups = set(readout_groups)
    print(f'readout_groups {readout_groups}  min {min(readout_groups)}')

    control_info = connect_info['body']['control']['0']['control_info']
    xpm_master   = control_info['xpm_master']
    pv_prefix    = control_info['pv_base']+':XPM:'+str(xpm_master)+':'

    teb = getattr(base['pcie'].DevPcie.Hsio.TimingRx.TriggerEventManager,f'TriggerEventBuffer[{lane}]')
    teb.Partition.set(min(readout_groups))
    teb.TriggerDelay.set(0)

    base['pcie'].StartRun()

    return apply_config(cfg, detsegm==0)

def apply_config(cfg,active):
    global pv_prefix
    rcfg = {}
    rcfg = cfg.copy()
    rcfg['user'] = {}
    rcfg['expert'] = {}

    linacMode = cfg['user']['LINAC']
    rcfg['user']['LINAC'] = linacMode
    rcfg['user']['Cu' if linacMode==0 else 'SC'] = {}

    pvdict  = {}  # dictionary of epics pv name : value
    for group in readout_groups:
        if linacMode == 0:   # Cu
            grp_prefix = 'group'+str(group)+'_eventcode'
            eventcode  = cfg['user']['Cu'][grp_prefix]
            rcfg['user']['Cu'][grp_prefix] = eventcode
            pvdict[str(group)+':L0Select'          ] = 2  # eventCode
            pvdict[str(group)+':L0Select_EventCode'] = eventcode
            pvdict[str(group)+':DstSelect'         ] = 1  # DontCare
        else:                # SC
            grp_prefix = 'group'+str(group)
            grp = cfg['user']['SC'][grp_prefix]
            rcfg['user']['SC'][grp_prefix] = grp
            pvdict[str(group)+':L0Select'          ] = grp['trigMode']
            pvdict[str(group)+':L0Select_FixedRate'] = grp['fixed']['rate']
            pvdict[str(group)+':L0Select_ACRate'   ] = grp['ac']['rate']
            pvdict[str(group)+':L0Select_EventCode'] = 0  # not an option
            pvdict[str(group)+':L0Select_Sequence' ] = grp['seq']['mode']
            pvdict[str(group)+':DstSelect'         ] = grp['destination']['select']

            # convert ac.ts0 through ac.ts5 to L0Select_ACTimeslot bitmask
            tsmask = 0
            for tsnum in range(6):
                tsval = grp['ac']['ts'+str(tsnum)]
                tsmask |= 1<<tsval
            pvdict[str(group)+':L0Select_ACTimeslot'] = tsmask

            # L0Select_SeqBit is one var used by all of seq.(burst/fixed/local)
            if grp['seq']['mode']==15: # burst
                seqbit = grp['seq']['burst']['mode']
            elif grp['seq']['mode']==16: # fixed rate
                seqbit = grp['seq']['fixed']['rate']
            elif grp['seq']['mode']==17: # local
                seqbit = grp['seq']['local']['rate']
            else:
                raise ValueError('Illegal value for trigger sequence mode')
            pvdict[str(group)+':L0Select_SeqBit'] = seqbit

            # DstSelect_Mask should come from destination.dest0 through dest15
            dstmask = 0
            for dstnum in range(16):
                dstval = grp['destination']['dest'+str(dstnum)]
                if dstval:
                    dstmask |= 1<<dstnum
            pvdict[str(group)+':DstSelect_Mask'] = dstmask

        grp_prefix = 'group'+str(group)
        grp = cfg['expert'][grp_prefix]
        rcfg['expert'][grp_prefix] = grp
        # 4 InhEnable/InhInterval/InhLimit
        for inhnum in range(4):
            pvdict[str(group)+':InhInterval'+str(inhnum)] = grp['inhibit'+str(inhnum)]['interval']
            pvdict[str(group)+':InhLimit'+str(inhnum)] = grp['inhibit'+str(inhnum)]['limit']
            pvdict[str(group)+':InhEnable'+str(inhnum)] = grp['inhibit'+str(inhnum)]['enable']

    names  = list(pvdict.keys())
    values = list(pvdict.values())
    names = [pv_prefix+'PART:'+n for n in names]

    # program the values
    ctxt = Context('pva')
    if active:
        ctxt.put(names,values)

    #  Capture firmware version for persistence in xtc
    #rcfg['firmwareVersion'] = ctxt.get(pv_prefix+'FwVersion').raw.value
    rcfg['firmwareBuild'  ] = ctxt.get(pv_prefix+'FwBuild').raw.value
    ctxt.close()

    return json.dumps(rcfg)

def apply_update(cfg):
    global pv_prefix

    rcfg = {}
    pvdict  = {}  # dictionary of epics pv name : value

    for key in cfg:
        if key == 'user':
            rcfg['user'] = {}

            linacMode = ocfg['user']['LINAC']  # this won't scan
            if full:
                rcfg['user']['LINAC'] = linacMode
            rcfg['user']['Cu' if linacMode==0 else 'SC'] = {}

            for group in readout_groups:
                if linacMode == 0:   # Cu
                    try:
                        grp_prefix = 'group'+str(group)+'_eventcode'
                        eventcode  = cfg['user']['Cu'][grp_prefix]
                        rcfg['user']['Cu'][grp_prefix] = eventcode
                        pvdict[str(group)+':L0Select'          ] = 2  # eventCode
                        pvdict[str(group)+':L0Select_EventCode'] = eventcode
                        pvdict[str(group)+':DstSelect'         ] = 1  # DontCare
                    except KeyError:
                        pass
                else:                # SC
                    pass   # nothing here to scan (too complicated to implement)

        if key == 'expert':
            rcfg['expert'] = {}
            for group in readout_groups:
                grp_prefix = 'group'+str(group)
                if grp_prefix in cfg['expert']:
                    grp = cfg['expert'][grp_prefix]
                    rcfg['expert'][grp_prefix] = {}
                    # 4 InhEnable/InhInterval/InhLimit
                    for inhnum in range(4):
                        inhkey = 'inhibit'+str(inhnum)
                        if inhkey in grp:
                            inhgrp = grp[inhkey]
                            rcfg['expert'][grp_prefix][inhkey] = inhgrp
                            rgrp = rcfg['expert'][grp_prefix][inhkey]
                            if 'interval' in inhgrp:
                                pvdict[str(group)+':InhInterval'+str(inhnum)] = inhgrp['interval']
                            if 'limit' in inhgrp:
                                pvdict[str(group)+':InhLimit'+   str(inhnum)] = inhgrp['limit']
                            if 'enable' in inhgrp:
                                pvdict[str(group)+':InhEnable'+  str(inhnum)] = inhgrp['enable']

        else:
            rcfg[key] = cfg[key]

    names  = list(pvdict.keys())
    values = list(pvdict.values())
    names = [pv_prefix+'PART:'+n for n in names]

    # program the values
    ctxt = Context('pva')
    ctxt.put(names,values)
    ctxt.close()

    return json.dumps(rcfg)

def tb_scan_keys(update):
    global ocfg
    #  extract updates
    cfg = {}
    copy_reconfig_keys(cfg,ocfg, json.loads(update))

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def tb_update(update):
    global ocfg
    #  extract updates
    cfg = {}
    update_config_entry(cfg,ocfg, json.loads(update))

    #  Apply config
    apply_update(cfg)

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def tb_unconfig():
    base['pcie'].StopRun()
    return base

def main():
    dev = '/dev/datadev_1'
    #
    #  Invoke generic root to lookup firmware type
    #
    with _DevRoot(dev = dev) as _base:
        fwImage = _base.DevPcie.AxiVersion.ImageName.get()
        _qsfp   = _base.DevPcie.Qsfp[0]
        rxpwr0  = (_qsfp.RxPwrRaw[0].get(),_qsfp.RxPwrRaw[1].get())
        txbias0 = (_qsfp.TxPwrRaw[0].get(),_qsfp.TxPwrRaw[1].get())
        _qsfp   = _base.DevPcie.Qsfp[1]
        rxpwr1  = (_qsfp.RxPwrRaw[0].get(),_qsfp.RxPwrRaw[1].get())
        txbias1 = (_qsfp.TxPwrRaw[0].get(),_qsfp.TxPwrRaw[1].get())

    print(f'Found image {fwImage}')
    print(f'RxPwr[0][0]: {rxpwr0}   RxPwr[1][0]: {rxpwr1}')
    print(f'TxBias[0][0]: {txbias0}   TxBias[1][0]: {txbias1}')

    #  Some images have conflicting versions of surf module
    sys.modules.pop('surf')


if __name__ == '__main__':
    main()

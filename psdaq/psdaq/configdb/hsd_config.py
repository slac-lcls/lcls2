from psdaq.utils import enable_l2si_drp
import l2si_drp
from psdaq.configdb.barrier import Barrier
from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from p4p.client.thread import Context
import os
import socket
import json
import time
import logging

ocfg = None
partitionDelay = None
epics_prefix = None
rawBuffSize = None
fexBuffSize = None
group = None

configVersion = [3,3,0]

barrier_global = Barrier()
args = {}

def supervisor_info(json_msg):
    nworker = 0
    supervisor=None
    mypid = os.getpid()
    myhostname = socket.gethostname()
    for drp in json_msg['body']['drp'].values():
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

def hsd_init(prefix, dev='dev/datadev_0'):
    global args
    global epics_prefix
    epics_prefix = prefix

    root = l2si_drp.DrpPgpIlvRoot(pollEn=False,devname=dev)
    root.__enter__()
    args['root'] = root.PcieControl.DevKcu1500
    args['core'] = root.PcieControl.DevKcu1500.AxiPcieCore.AxiVersion.DRIVER_TYPE_ID_G.get()==0

    hsd_unconfig(prefix)

def hsd_connect(msg):

    root = args['root']

    alloc_json = json.loads(msg)
    supervisor,nworker = supervisor_info(alloc_json)
    barrier_global.init(supervisor,nworker)

    if barrier_global.supervisor:
        # Check clock programming
        clockrange = (180.,190.)
        rate = root.MigIlvToPcieDma.MonClkRate_3.get()*1.e-6

        if (rate < clockrange[0] or rate > clockrange[1]):
            logging.info(f'Si570 clock rate {rate}.  Reprogramming')
            root.I2CBus.programSi570(1300/7.)

    barrier_global.wait()

    time.sleep(1)

    # Set linkId
    
    hostname = socket.gethostname()
    ipaddr   = socket.gethostbyname(hostname).split('.')
    linkId   = 0xfb000000 | (int(ipaddr[2])<<8) | (int(ipaddr[3])<<0)
    if not args['core']:
        linkId |= 0x40000

    for i in range(4):
        getattr(root,f'TxLinkId[{i}]').set(linkId | i<<16)

    # Retrieve connection information from EPICS
    # May need to wait for other processes here {PVA Server, hsdioc}, so poll
    ctxt = Context('pva')
    for i in range(50):
        values = ctxt.get(epics_prefix+':PADDR_U')
        if values!=0:
            break
        print('{:} is zero, retry'.format(epics_prefix+':PADDR_U'))
        time.sleep(0.1)

    #  validate linkId: EPICS returns linkId as a signed int32
    remoteLinkId = ctxt.get(epics_prefix+':MONPGP').remlinkid[0]
    match = (remoteLinkId^linkId)&0xffffffff
    if match:
        raise ValueError(f'pgpTxLinkId [{linkId:x}] does not match remoteLinkId [{remoteLinkId:x}] from EPICS (match={match:x})')   

    ctxt.close()

    d = {}
    d['paddr'] = values
    return d

def hsd_config(connect_str,prefix,cfgtype,detname,detsegm,rog):
    global partitionDelay
    global rawBuffSize
    global fexBuffSize
    global ocfg
    global group

    group = rog

    root = args['root']

    #  Some diagnostic printout
    def rxcnt(lane,name):
        return getattr(getattr(root,f'Pgp3AxiL[{lane}]'),name).get()

    def print_field(name):
        logging.info(f'{name:15s}: {rxcnt(0,name):04x} {rxcnt(1,name):04x} {rxcnt(2,name):04x} {rxcnt(3,name):04x}')

    print_field('RxFrameCount')
    print_field('RxFrameErrorCount')

    def toggle(var,value):
        var.set(value)
        time.sleep(10.e-6)
        var.set(0)

    #  Reset the PGP links
    toggle(root.MigIlvToPcieDma.UserReset,1)
    #  QPLL reset
    toggle(root.PgpQPllReset,1)
    #  Tx reset
    toggle(root.PgpTxReset,1)
    #  Rx reset
    toggle(root.PgpRxReset,1)

    #  On to the business of configure
    ctxt = Context('pva')

    cfg = get_config(connect_str,cfgtype,detname,detsegm)
    algVsn = cfg['alg:RO']['version:RO']

    if algVsn != configVersion:
        raise RuntimeError(f'configdb version {algVsn} does not match software required version {configVersion}')
    
    # program the group
    expert = cfg['expert']
    expert['readoutGroup'] = group
    expert['enable'   ] = 1  # Need to enable to get buffer sizes
    apply_config(ctxt,cfg)

    # fetch the current configuration for defaults not specified in the configuration
    values = ctxt.get(epics_prefix+':CONFIG')

    # Wait for the L0Delay to update
    while True:
        monTiming = ctxt.get(epics_prefix+':MONTIMING')
        if monTiming.group == group:
            break
        print(f'Polling monTiming: group {monTiming.group}/{group}')
        time.sleep(0.2)

    print(epics_prefix+':MONTIMING')
    print(monTiming)

    # fetch the xpm delay
    partitionDelay = monTiming.l0delay
    print('partitionDelay {:}'.format(partitionDelay))

    # fetch the freesz
    rawBuffSize = ctxt.get(epics_prefix+':MONRAWBUF').freesz
    fexBuffSize = ctxt.get(epics_prefix+':MONFEXBUF').freesz

    ocfg = cfg
    user_to_expert(cfg)

    # overwrite expert fields from user input
    raw = cfg['user']['raw']
    fex = cfg['user']['fex']
    expert = cfg['expert']
    expert['readoutGroup'] = group
    expert['enable'   ] = 1
    expert['raw_prescale'] = raw['prescale']
    if 'keep' in raw:
        expert['raw_keep']  = raw['keep']
    else:
        expert['raw_keep'] = 0
        logging.warning('No user.raw.keep entry in config.  Run hsd_config_update.py')

    fex_xpre       = int((fex['xpre' ]+3)/4)
    fex_xpost      = int((fex['xpost']+3)/4)
    keepRows = ctxt.get(epics_prefix+':KEEPROWS').value
    if keepRows is None:
        raise RuntimeException('Unable to get KEEPROWS')
    if keepRows == 0:
        logging.warning('Firmware version doesnt support KEEPROWS checking')
    else:
        if not (fex_xpost < keepRows*10):
            raise ValueError(f'xpost {fex_xpost} must be less than {keepRows*40}')
        if not (fex_xpre < keepRows*10):
            raise ValueError(f'xpost {fex_xpre} must be less than {keepRows*40}')

    expert['fex_xpre' ] = fex_xpre
    expert['fex_xpost'] = fex_xpost
    if 'dymin' in fex:
        expert['fex_ymin' ] = fex['corr']['baseline']+fex['dymin']
        expert['fex_ymax' ] = fex['corr']['baseline']+fex['dymax']
    else:
        expert['fex_ymin' ] = fex['ymin']
        expert['fex_ymax' ] = fex['ymax']
    expert['fex_prescale'] = fex['prescale']

    # program the values
    apply_config(ctxt,cfg)

    # clear jesd error latches
    rst = ctxt.get(epics_prefix+':RESET')
    rst['jesdclear'] = 1
    ctxt.put(epics_prefix+':RESET',rst,wait=True)
    rst['jesdclear'] = 0
    ctxt.put(epics_prefix+':RESET',rst,wait=False)
    
    fwver = ctxt.get(epics_prefix+':FWVERSION').value
    fwbld = ctxt.get(epics_prefix+':FWBUILD'  ).value
    cfg['firmwareVersion'] = fwver
    cfg['firmwareBuild'  ] = fwbld
    print(f'fwver: {fwver}')
    print(f'fwbld: {fwbld}')

    ctxt.close()

    ocfg = cfg
    return json.dumps(cfg)

def hsd_unconfig(prefix):
    global epics_prefix
    epics_prefix = prefix
    
    ctxt = Context('pva')
    
    def epics_unconfig(pvname):
        valuesA = ctxt.get(pvname+':CONFIG')
        if valuesA['enable'] ==1 :
            valuesA['enable'] = 0
            print(pvname)
            ctxt.put(pvname+':CONFIG',valuesA,wait=True)

            #  This handshake seems to be necessary, or at least the .get()
            complete = False
            for i in range(100):
                complete = ctxt.get(pvname+':READY')!=0
                if complete: break
                print('hsd_unconfig wait for complete',i)
                time.sleep(0.1)
            if complete:
                print('hsd unconfig complete')
            else:
                raise Exception('timed out waiting for hsd_unconfig')
        else:
            print(f'{pvname}: enable already false')
    
    # disable both A and B detectors, so we don't get unwanted deadtime
    # from a detector not in the partition.
    epics_unconfig(prefix[:-1]+"A")
    epics_unconfig(prefix[:-1]+"B")

    ctxt.close()

    return None;

def user_to_expert(cfg):
    global group
    global ocfg

    d = {}
    hasUser = 'user' in cfg
    if hasUser:
        raw_start = None
        raw_gate  = None
        fex_start = None
        fex_gate  = None

        hasRaw = 'raw' in cfg['user']
        raw = cfg['user']['raw']
        if (hasRaw and 'start_ns' in raw):
            raw_start      = int((raw['start_ns']*1300/7000 - partitionDelay*200)*160/200)

            if raw_start < 0:
                print('partitionDelay {:}  raw_start_ns {:}  raw_start {:}'.
                      format(partitionDelay,raw['start_ns'],raw_start))
                raise ValueError('raw_start is too small by {:} ns'.
                                 format(-raw_start/0.16*14./13))
            if raw_start > 0x3fff:
                print('partitionDelay {:}  raw_start_ns {:}  raw_start {:}'.
                      format(partitionDelay,raw['start_ns'],raw_start))
                raise ValueError('start_ns is too large by {:} ns'.
                                 format((raw_start-0x3fff)/0.16*14./13))

            d['expert.raw_start'] = raw_start

        if (hasRaw and 'gate_ns' in raw):
            raw_gate     = int(raw['gate_ns']*0.160*13/14) # in "160" MHz clks
            raw_nsamples = raw_gate*40
            # raw_gate register is 14 bits
            if raw_gate < 0:
                raise ValueError('raw_gate computes to < 0')
            if raw_gate > rawBuffSize:
                raise ValueError(f'raw_gate ({raw_gate}/{40*raw_gate}sam) computes to > rawBuffSize ({rawBuffSize})')

            d['expert.raw_gate'] = raw_gate

        hasFex = 'fex' in cfg['user']
        fex = cfg['user']['fex']
        if (hasFex and 'start_ns' in fex):
            fex_start      = int((fex['start_ns']*1300/7000 - partitionDelay*200)*160/200)

            if fex_start < 0:
                print('partitionDelay {:}  fex_start_ns {:}  fex_start {:}'.
                      format(partitionDelay,fex['start_ns'],fex_start))
                raise ValueError('fex_start is too small by {:} ns'.
                                 format(-fex_start/0.16*14./13))
            if fex_start > 0x3fff:
                print('partitionDelay {:}  fex_start_ns {:}  fex_start {:}'.
                      format(partitionDelay,fex['start_ns'],fex_start))
                raise ValueError('start_ns is too large by {:} ns'.
                                 format((fex_start-0x3fff)/0.16*14./13))

            d['expert.fex_start'] = fex_start

        if (hasFex and 'gate_ns' in fex):
            fex_gate     = int(fex['gate_ns']*0.160*13/14) # in "160" MHz clks
            fex_nsamples = fex_gate*40
            # fex_gate register is 14 bits
            if fex_gate < 0:
                raise ValueError('fex_gate computes to < 0')
            if fex_gate > 4000:
                raise ValueError('fex_gate computes to > 4000; fex_nsamples > 160000')

            d['expert.fex_gate'] = fex_gate

        #  Check the deadtime watermarks
        full_rtt = cfg['expert']['full_rtt']
        full_evt = cfg['expert']['full_event']
        if raw_start and raw_gate:
            full_size = (160*full_rtt)//200 + raw_start + raw_gate
            if full_size > rawBuffSize:
                low_rate_size = (160*full_rtt)//200 + raw_gate
                logging.warning(f'Raw full threshold ({full_size}) computes to > raw full size ({rawBuffSize}).  Lowering to {low_rate_size}.')
                full_size = low_rate_size
            d['expert.full_size_raw'] = full_size
            evt = int(raw_start/160 + full_rtt/200)
            if evt > full_evt:
                logging.warning(f'full_event threshold protects raw buffers up to {full_evt/evt} MHz.  Set full_event > {evt} for MHz running or increase group {group} L0Delay by {evt-full_evt}.')

        if fex_start and fex_gate:
            full_size = (160*full_rtt)//200 + fex_start + fex_gate
            if full_size > fexBuffSize:
                low_rate_size = (160*full_rtt)//200 + fex_gate
                logging.warning(f'Fex full threshold ({full_size}) computes to > fex full size ({fexBuffSize}).  Lowering to {low_rate_size}.')
                full_size = low_rate_size
            d['expert.full_size_fex'] = full_size
            evt = int(fex_start/160 + full_rtt/200)
            if evt > full_evt:
                logging.warning(f'full_event threshold protects fex buffers up to {full_evt/evt} MHz.  Set full_event > {evt} for MHz running or increase group {group} L0Delay by {evt-full_evt}.')

    update_config_entry(cfg,ocfg,d)

def apply_config(ctxt,cfg):
    global epics_prefix

    # program the values
    print(epics_prefix)
    ctxt.put(epics_prefix+':READY',0,wait=True)
    if 'adccal' in cfg:
        values = ctxt.get(epics_prefix+':ADCCAL')
        for k,v in cfg['adccal'].items():
            values[k] = v
        ctxt.put(epics_prefix+':ADCCAL',values,wait=True)
    values = ctxt.get(epics_prefix+':CONFIG')
    if 'expert' in cfg:
        xsec = set(values.keys()) & set(cfg['expert'].keys())
        for k in xsec:
            values[k] = cfg['expert'][k]
    values['fex_corr_baseline'] = cfg['user']['fex']['corr']['baseline']
    values['fex_corr_accum'   ] = cfg['user']['fex']['corr']['accum']
    ctxt.put(epics_prefix+':CONFIG',values,wait=True)

    # the completion of the "put" guarantees that all of the above
    # have completed (although in no particular order)
    complete = False
    for i in range(100):
        complete = ctxt.get(epics_prefix+':READY')!=0
        if complete: break
        print('hsd config wait for complete',i)
        time.sleep(0.1)
    if complete:
        print('hsd config complete')
        time.sleep(2)
        print('hsd config returning')
    else:
        raise Exception('timed out waiting for hsd configure')


def hsd_scan_keys(update):
    global ocfg
    print('hsd_scan_keys update {}'.format(update))
    print('hsd_scan_keys ocfg {}'.format(ocfg))
    #  extract updates
    cfg = {}
    copy_reconfig_keys(cfg, ocfg, json.loads(update))
    #  Apply group
    user_to_expert(cfg)
    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)

def hsd_update(update):
    global ocfg
    #  extract updates
    cfg = {}
    update_config_entry(cfg,ocfg, json.loads(update))
    #  Apply group
    user_to_expert(cfg)
    #  Apply config
    ctxt = Context('pva')
    apply_config(ctxt,cfg)
    ctxt.close()

    #  Retain mandatory fields for XTC translation
    for key in ('detType:RO','detName:RO','detId:RO','doc:RO','alg:RO'):
        copy_config_entry(cfg,ocfg,key)
        copy_config_entry(cfg[':types:'],ocfg[':types:'],key)
    return json.dumps(cfg)


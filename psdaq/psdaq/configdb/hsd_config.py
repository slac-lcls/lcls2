from psdaq.utils import enable_l2si_drp
import l2si_drp
from psdaq.configdb.barrier import *
from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from p4p.client.thread import Context
import json
import time
import logging
import datetime

ocfg = None
partitionDelay = None
epics_prefix = None
rawBuffSize = None
fexBuffSize = None
group = None

configVersion = [4,0,0]

barrier_global = Barrier()
args = {}

def hsd_init(prefix, dev='dev/datadev_0'):
    global args
    global epics_prefix
    epics_prefix = prefix

    args['dev'] = dev
    
    if True:   # Until SUBMODULES is updated
        root = l2si_drp.DrpPgpIlvRoot(pollEn=False,devname=dev)
        root.__enter__()
        args['root'] = root.PcieControl.DevPcie
        args['core'] = root.PcieControl.DevPcie.AxiPcieCore.AxiVersion.DRIVER_TYPE_ID_G.get()==0
        args['swclk'] = args['core']

    else:
        #  Lookup the board type
        boardType = 'Kcu1500'
        tst  = l2si_drp.DrpPgpIlvRoot(pollEn=False,devname=dev,boardType=boardType)
        tst.start()
        imageName = tst.PcieControl.DevPcie.AxiPcieCore.AxiVersion.ImageName.get()
        if 'C1100' in imageName:
            boardType = 'VariumC1100'
        deviceId = tst.PcieControl.DevPcie.AxiPcieCore.AxiVersion.DeviceId.get()
        logging.warning(f'Found boardType {boardType} and deviceId {deviceId}')
        tst.stop()

        root = l2si_drp.DrpPgpIlvRoot(pollEn=False,devname=dev,boardType=boardType,extended=deviceId!=0)
        root.__enter__()
        args['root'] = root.PcieControl.DevPcie
        args['core'] = deviceId==0
        args['swclk'] = boardType=='Kcu1500' and deviceId==0
    
    hsd_unconfig(prefix)

def hsd_connect(msg):

    root = args['root']

    alloc_json = json.loads(msg)
    supervisor,nworker = supervisor_info(alloc_json,args['dev'])
    print(f'hsd_connect: supervisor [{supervisor}] nworker [{nworker}]')
    
    barrier_global.init(supervisor,nworker)

    if barrier_global.supervisor and args['swclk']:
        # Check clock programming
        clockrange = (180.,190.)
        rate = root.MigIlvToPcieDma.MonClkRate_3.get()*1.e-6
        print(f'hsd_connect: clock rate [{rate}]')
        
        if (rate < clockrange[0] or rate > clockrange[1]):
            logging.warning(f'Si570 clock rate {rate}.  Reprogramming')
            root.I2CBus.programSi570(1300/7.)

            time.sleep(1.0)
            rate = root.MigIlvToPcieDma.MonClkRate_3.get()*1.e-6

            if (rate < clockrange[0] or rate > clockrange[1]):
                logging.error(f'Si570 clock programming failed.  rate {rate}.')

    barrier_global.wait()

    def toggleReset(v):
        v.set(1)
        time.sleep(0.001)
        v.set(0)
        time.sleep(0.001)

    if root.PgpQPllLock.get()==0:
        logging.warning('QPLL unlocked.  Resetting...')
        toggleReset(root.PgpQPllReset)
        time.sleep(0.001)
        if root.PgpQPllLock.get()==0:
            logging.error(f'PGP QPLL failed to lock')

        toggleReset(root.PgpTxReset)
        toggleReset(root.PgpRxReset)
    
    time.sleep(1)

    # Set linkId
    
    hostname = socket.gethostname()
    ipaddr   = socket.gethostbyname(hostname).split('.')
    linkId   = 0xfb000000 | (int(ipaddr[2])<<8) | (int(ipaddr[3])<<0)
    if not args['core']:
        linkId |= 0x40000

    for i in range(4):
        getattr(root,f'TxLinkId[{i}]').set(linkId | i<<16)

    #
    # Check that the hsdioc process is alive
    #
    if True:
        ctxt = Context('pva',nt=False)
        seconds = ctxt.get(epics_prefix+':FEXOOR').timeStamp.secondsPastEpoch
        dt  = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        diff = now-dt
        diff_s = diff.total_seconds()
        print(f'FEXOOR latency is {diff_s} seconds')
        if diff_s > 100:
            raise ValueError(f'hsdioc process may be dead.')
    
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
    insBuffSize = ctxt.get(epics_prefix+':MONINSBUF').freesz
    print(f'rawBuffSize {rawBuffSize}  fexBuffSize {fexBuffSize}  insBuffSize {insBuffSize}')
    
    ocfg = cfg
    user_to_expert(cfg)

    # overwrite expert fields from user input
    raw = cfg['user']['raw']
    fex = cfg['user']['fex']
    ins = cfg['user']['inspect']
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

    expert['inspect_prescale'] = ins['prescale']
    
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
    print(f'fwver: {fwver:x}')
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
        ins_start = None
        ins_gate  = None

        full_rtt = cfg['expert']['full_rtt']
        full_evt = cfg['expert']['full_event']
        
        def _check_start_and_gate(stream, buffSize, sparse):
            hasStream = stream in cfg['user']
            cfg_s = cfg['user'][stream]
            start = None
            gate  = None
            #  Check the start
            if (hasStream and 'start_ns' in cfg_s):
                start      = int((cfg_s['start_ns']*1300/7000 - partitionDelay*200)*160/200)
                # start register is 20 bits
                if start < 0:
                    print(f'partitionDelay {partitionDelay}  {stream}_start_ns {cfg_s["start_ns"]}  {stream}_start {start}')
                    raise ValueError(f'{stream}_start is too small by {-start/0.16*14./13} ns')
                if start > 0xfffff:
                    print(f'partitionDelay {partitionDelay}  {stream}_start_ns {cfg_s["start_ns"]}  {stream}_start {start}')
                    raise ValueError(f'{stream}_start_ns is too large by {start-0xfffff)/0.16*14./13} ns')

            d[f'expert.{stream}_start'] = start

            #  Check the gate
            if (hasStream and 'gate_ns' in cfg_s):
                gate     = int(cfg_s['gate_ns']*0.160*13/14) # in "160" MHz clks
                nsamples = gate*40
                # gate register is 20 bits
                if gate < 0:
                    raise ValueError(f'{stream}_gate computes to < 0')
                if gate > buffSize:
                    if sparse:
                        logging.warning(f'{stream}_gate ({gate}/{40*gate}sam) computes to > {stream}BuffSize ({buffSize})')
                    else:
                        raise ValueError(f'{stream}_gate ({gate}/{40*gate}sam) computes to > {stream}BuffSize ({buffSize})')
                if gate > 0xfffff:
                    raise ValueError(f'{stream}_gate ({gate}/{40*gate}sam) computes to > 20 bits')
                    
            d[f'expert.{stream}_gate'] = gate

            #  Check the deadtime watermarks
            if start and gate:
                full_size = (160*full_rtt)//200 + start + gate
                if full_size > buffSize:
                    low_rate_size = (160*full_rtt)//200 + gate
                    logging.warning(f'{stream} full threshold ({full_size}) computes to > {stream} full size ({buffSize}).  Lowering to {low_rate_size}.')
                    full_size = low_rate_size
                d['expert.full_size_{stream}'] = full_size
                evt = int(start/160 + full_rtt/200)
                if evt > full_evt:
                    logging.warning(f'full_event threshold protects {stream} buffers up to {full_evt/evt} MHz.  Set full_event > {evt} for MHz running or increase group {group} L0Delay by {evt-full_evt}.')

            
        _check_start_and_gate ('raw'    ,rawBuffSize, False)
        _check_start_and_gate ('fex'    ,fexBuffSize, True)
        _check_start_and_gate ('inspect',insBuffSize, False)
        
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


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='test connect method')
    parser.add_argument('-P', default='DAQ:XPP:HSD:1_01:A', metavar='PREFIX')
    parser.add_argument('--dev', default='/dev/datadev_1', help='device name')
    pargs = parser.parse_args()

    hsd_init(pargs.P, dev=pargs.dev)

    #  For supervisor pattern
    alloc = {'body': {'drp' : {'1': {'active'   : 1,
                                     'proc_info': {'host':socket.gethostname(),
                                                   'pid' : os.getpid()}}}}}
    hsd_connect(json.dumps(alloc))
    print(f'connect complete')

    #  To lookup configuration
    conn = {'body': {'control' : {'0': {'active': 1,
                                        'control_info': {'cfg_dbase': 'https://psdmint.sdf.slac.stanford.edu/ws-auth/configdb/ws/configDB',
                                                         'instrument': 'xpp',
                                                         'pv_base': 'DAQ:FEH',
                                                         'slow_update_rate': 1,
                                                         'xpm_master': 4}}}}}
    print(f'conn {conn}')
    
    hsd_config(json.dumps(conn),pargs.P,'BEAM','hsd',2,0)
    

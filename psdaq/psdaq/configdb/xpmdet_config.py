import DrpTDet
from psdaq.configdb.barrier import Barrier
from psdaq.cas.xpm_utils import timTxId
import os
import socket
import rogue
import time
import json
import logging

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

def xpmdet_init(dev='/dev/datadev_0',lanemask=1,timebase="186M",verbosity=0):
    global args

    logging.info('xpmdet_init')

    args["timebase"]=timebase
    args["lanemask"]=lanemask

    root = DrpTDet.DrpTDetRoot(pollEn=False,devname=dev)
    root.__enter__()

    logging.info('Reset timing data path')
    root.PcieControl.DevKcu1500.TDetTiming.TimingFrameRx.C_RxReset()
    time.sleep(0.1)

    args['root'] = root.PcieControl.DevKcu1500
    return root

# called on alloc
def xpmdet_connectionInfo(alloc_json_str):
    root = args['root']

    xma = root.TDetTiming.TriggerEventManager.XpmMessageAligner

    alloc_json = json.loads(alloc_json_str)
    supervisor,nworker = supervisor_info(alloc_json)
    barrier_global.init(supervisor,nworker)

    if barrier_global.supervisor:
        
        if args["timebase"]=="186M":
            clockrange = (180.,190.)
        elif args["timebase"]=="119M":
            clockrange = (115.,125.)
        else:
            clockrange = None

        if clockrange is not None:
            if root.AxiPcieCore.AxiVersion.DRIVER_TYPE_ID_G.get()==0:
                # check timing reference clock, program if necessary
                rate = root.TDetTiming.refClockRate()

                if (rate < clockrange[0] or rate > clockrange[1]):
                    root.I2CBus.programSi570(args["timebase"])
            else:
                logging.warning('Supervisor is not I2cBus manager')

        txId = timTxId('tdet')
        xma.TxId.set(txId)

        #  Disable all timing links
        for i in range(8):
            teb = getattr(root.TDetTiming.TriggerEventManager,f'TriggerEventBuffer[{i}]')
            teb.MasterEnable.set(0)
            teb.ResetCounters()
            teb.FifoReset()

    barrier_global.wait()

    xpmdet_unconfig()

    rxId = xma.RxId.get()
    logging.info('rxId {:x}'.format(rxId))
        
    if (rxId==0 or rxId==0xffffffff or (rxId&0xff)>15):
        logging.critical(f"XPM Remote link id register illegal value: 0x{rxId:08x}. Trying RxPllReset.");
        tim = root.TDetTiming.TimingFrameRx
        tim.RxPllReset.set(1)
        tim.RxPllReset.set(0)
        time.sleep(0.0001)
        tim.C_RxReset()
        time.sleep(0.1)

        rxId = xma.RxId.get()
        logging.info('rxId {:x}'.format(rxId))

        if (rxId==0 or rxId==0xffffffff or (rxId&0xff)>15):
            logging.critical(f"XPM Remote link id register illegal value: 0x{rxId:08x}. Aborting.  Try TxPllReset.");
            raise RuntimeError(f"Illegal XPM Remote link id. Try TxPllReset.")

    connect_info = {}
    connect_info['paddr'] = rxId

    return connect_info

#  Apply the full configuration
def xpmdet_connect(grp,length):
    root = args['root']

    lm = args["lanemask"]
    for i in range(8):
        if (lm & (1<<i)):
            teb = getattr(root.TDetTiming.TriggerEventManager,f'TriggerEventBuffer[{i}]')
            teb.ResetCounters()
            teb.PauseThreshold.set(16)
            teb.Partition.set(grp)
            teb.TriggerSource.set(0)
            teb.TriggerDelay.set(0)
            teb.MasterEnable.set(True)

            getattr(root.TDetSemi,f'Clear_{i&3}').set(1)
            getattr(root.TDetSemi,f'Length_{i&3}').set(length)
            getattr(root.TDetSemi,f'Clear_{i&3}').set(0)
            getattr(root.TDetSemi,f'Enable_{i&3}').set(1)

def xpmdet_unconfig():
    root = args['root']

    #  Clear TDetSemi
    lm = args["lanemask"]
    for i in range(8):
        if (lm & (1<<i)):
            getattr(root.TDetSemi,f'Enable_{i}').set(0)
            getattr(root.TDetSemi,f'Clear_{i}').set(1)

    return root

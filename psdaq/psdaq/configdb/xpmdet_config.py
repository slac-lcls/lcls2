from psdaq.utils import enable_l2si_drp
import l2si_drp
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
#logging.basicConfig(level=logging.INFO)

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


def detect_C1100():
    ''' Detect if the board is a C1100 by reading /proc/datadev_0 '''
    file_datadev='/proc/datadev_0'
    isC1100 = False
    try:
        with open(file_datadev, 'r', encoding='utf-8') as file:
            for line in file:
                if 'Build String' in line:
                    isC1100 = 'C1100' in line
                    break
        return isC1100

    except FileNotFoundError:
        logging.error(f"Error: File '{file_datadev}' not found.")
        return False
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return False
    
def dumpTiming(tim):
    logging.warning(f'FidCount  : {tim.FidCount.get()}')
    logging.warning(f'RxRstCount: {tim.RxRstCount.get()}')
    logging.warning(f'RxDecErrs : {tim.RxDecErrCount.get()}')
    logging.warning(f'RxDspErrs : {tim.RxDspErrCount.get()}')

def xpmdet_init(dev='/dev/datadev_0',lanemask=1,timebase="186M",verbosity=0):
    global args
    logging.info('xpmdet_init')

    args["timebase"]=timebase
    args["lanemask"]=lanemask
    
    if (detect_C1100()):
       # print("Board Detected C1100")
        root = l2si_drp.DrpTDetRoot(pollEn=False,devname=dev,boardType='VariumC1100',qsa=False, xvcPort=None)
        root.__enter__()
        logging.info("Board Detected C1100")
        
    else:
       # print("Board Detected KCU1500")
        root = l2si_drp.DrpTDetRoot(pollEn=False,device=dev)
        root.__enter__()
        logging.info("Board Detected KCU1500")

    args['root'] = root.PcieControl.DevPcie
    args['core'] = root.PcieControl.DevPcie.AxiPcieCore.AxiVersion.DRIVER_TYPE_ID_G.get()==0
#    print("init done")
##  Moved to connectionInfo so supervisor can execute it only once
#    logging.info('Reset timing data path')
#    dumpTiming(root.PcieControl.DevKcu1500.TDetTiming.TimingFrameRx)
#    root.PcieControl.DevKcu1500.TDetTiming.TimingFrameRx.C_RxReset()
#    time.sleep(0.1)
#    root.PcieControl.DevKcu1500.TDetTiming.TimingFrameRx.ClearRxCounters()

            
    return root

# called on alloc
def xpmdet_connectionInfo(alloc_json_str):
   # print("xpmdet_connectionInfo")
    root = args['root']

    xma = root.TDetTiming.TriggerEventManager.XpmMessageAligner
   # time.sleep(1)

    alloc_json = json.loads(alloc_json_str)
    supervisor,nworker = supervisor_info(alloc_json)
    #print(f"Am I supervisor ? {supervisor} {nworker}")
    barrier_global.init(supervisor,nworker)

    if barrier_global.supervisor:
        tim = root.TDetTiming.TimingFrameRx
        dumpTiming(tim)
        time.sleep(0.1)
        tim.ClearRxCounters()

        if args["timebase"]=="186M":
            clockrange = (180.,190.)
        elif args["timebase"]=="119M":
            clockrange = (115.,125.)
        else:
            clockrange = None

        if clockrange is not None:
            if True:
#            if args['core']:
                # check timing reference clock, program if necessary
                rate = root.TDetTiming.refClockRate()

                if (rate < clockrange[0] or rate > clockrange[1]):
                    root.I2CBus.programSi570(119. if args["timebase"]=="119M" else 1300/7.)
                    tim.RxPllReset.set(1)
                    tim.RxPllReset.set(0)
                    time.sleep(0.0001)
                    dumpTiming(tim)
                    tim.C_RxReset()
                    time.sleep(0.1)
                    tim.ClearRxCounters()
            else:
                logging.warning('Supervisor is not I2cBus manager')

        txId = timTxId('tdet')
        xma.TxId.set(txId)

        rxId = xma.RxId.get()
        logging.info('rxId {:x}'.format(rxId))

        #  Disable all timing links
        for i in range(8):
            teb = getattr(root.TDetTiming.TriggerEventManager,f'TriggerEventBuffer[{i}]')
            teb.MasterEnable.set(0)
            teb.ResetCounters()
            teb.FifoReset()
        xpmdet_unconfig()
    
        logging.info('unconfig Initial rxId {:x}'.format(rxId))

        rxId = xma.RxId.get()
        logging.info('rxId {:x}'.format(rxId))

  
        if (rxId==0 or rxId==0xffffffff or (rxId&0xff)>15):
            logging.warning(f"XPM Remote link id register illegal value: 0x{rxId:08x}. Trying RxPllReset.");
            tim = root.TDetTiming.TimingFrameRx
            tim.RxPllReset.set(1)
            tim.RxPllReset.set(0)
            time.sleep(0.0001)
            dumpTiming(tim)
            tim.C_RxReset()
            time.sleep(1.0)
            tim.ClearRxCounters()

            rxId = xma.RxId.get()
            if (rxId==0 or rxId==0xffffffff or (rxId&0xff)>15):
                logging.critical(f"XPM Remote link id register illegal value: 0x{rxId:08x}. Aborting.  Try TxPllReset.");
                raise RuntimeError(f"Illegal XPM Remote link id. Try TxPllReset.")
    barrier_global.wait()
    rxId = xma.RxId.get()
    logging.info('rxId {:x}'.format(rxId))

    connect_info = {}
    connect_info['paddr'] = rxId

    return connect_info

# called on dealloc
def xpmdet_connectionShutdown():
    barrier_global.shutdown()
    return True

#  Apply the full configuration
def xpmdet_connect(grp,length):
    root = args['root']

    lm = args["lanemask"]
    for i in range(4):
        if (lm & (1<<i)):
            il = i if args['core'] else i+4
            teb = getattr(root.TDetTiming.TriggerEventManager,f'TriggerEventBuffer[{il}]')
            teb.ResetCounters()
            teb.PauseThreshold.set(16)
            teb.Partition.set(grp)
            teb.TriggerSource.set(0)
            teb.TriggerDelay.set(0)
            teb.MasterEnable.set(True)

            getattr(root.TDetSemi,f'Clear_{i}').set(1)
            getattr(root.TDetSemi,f'Length_{i}').set(length)
            getattr(root.TDetSemi,f'Clear_{i}').set(0)
            getattr(root.TDetSemi,f'Enable_{i}').set(1)

    return True

def xpmdet_unconfig():
    root = args['root']

    #  Clear TDetSemi
    lm = args["lanemask"]
    for i in range(4):
        if (lm & (1<<i)):
            getattr(root.TDetSemi,f'Enable_{i}').set(0)
            getattr(root.TDetSemi,f'Clear_{i}').set(1)

    return root

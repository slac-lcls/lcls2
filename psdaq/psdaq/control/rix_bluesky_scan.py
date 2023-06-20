# rix_bluesky_scan.py

from bluesky import RunEngine
import sys
import logging
import threading
import time
from psdaq.control.ControlDef import ControlDef
from psdaq.control.DaqControl import DaqControl
from psdaq.control.BlueskyScan import BlueskyScan
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, choices=range(0, 8), default=2,
                    help='platform (default 2)')
parser.add_argument('-C', metavar='COLLECT_HOST', default='drp-neh-ctl001',
                    help='collection host (default drp-neh-ctl001)')
parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                    help='timeout msec (default 10000)')
parser.add_argument('-c', type=int, metavar='READOUT_COUNT', default=120, help='# of events to aquire at each step (default 120)')
parser.add_argument('-g', type=int, metavar='GROUP_MASK', default=36, help='bit mask of readout groups (default 36)')
parser.add_argument('--groups', type=int, nargs='+', metavar='GROUP_LIST', default=[], help='list of readout groups (overrides -g)')
parser.add_argument('--config', metavar='ALIAS', default='BEAM', help='configuration alias (default BEAM)')
parser.add_argument('--detname', default='scan', help="detector name (default 'scan')")
parser.add_argument('--scantype', default='scan', help="scan type (default 'scan')")
parser.add_argument('--seqctl' , default=None, type=str, nargs='+', help="sequence control (e.g. DAQ:NEH:XPM:0:SeqReset 4 [DAQ:NEH:XPM:0:SeqDone])")
parser.add_argument('--record', action='store_true', help='enable recording of data')
parser.add_argument('-v', action='store_true', help='be verbose')
args = parser.parse_args()

if len(args.groups) > 0:
    g = 0
    for i in args.groups:
        g += 1<<i
    args.g = g

if args.g is not None:
    if args.g < 1 or args.g > 255:
        parser.error('readout group mask (-g) must be 1-255')

if args.c < 1:
    parser.error('readout count (-c) must be >= 1')

if args.seqctl is not None and (len(args.seqctl)<2 or len(args.seqctl)>3):
    parser.error('seqctl must have 2 or 3 argments')

# instantiate DaqControl object
control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

# configure logging handlers
if args.v:
    print(f'Trying to connect with DAQ at host {args.C} platform {args.p}...')
instrument = control.getInstrument()
if instrument is None:
    print(f'Failed to connect with DAQ at host {args.C} platform {args.p}')
    sys.exit(1)

if args.v:
    level=logging.DEBUG
else:
    level=logging.WARNING
logging.basicConfig(level=level)
logging.info('logging initialized')

# get initial DAQ state
daqState = control.getState()
logging.info('initial state: %s' % daqState)
if daqState == 'error':
    sys.exit(1)

# optionally set BEAM or NOBEAM
if args.config is not None:
    # config alias request
    rv = control.setConfig(args.config)
    if rv is not None:
        logging.error('%s' % rv)

RE = RunEngine({})

# cpo thinks this is more for printout of each step
from bluesky.callbacks.best_effort import BestEffortCallback
bec = BestEffortCallback()

# Send all metadata/data captured to the BestEffortCallback.
RE.subscribe(bec)

from ophyd.sim import motor1
from bluesky.plans import scan

# instantiate BlueskyScan object
mydaq = BlueskyScan(control, daqState=daqState)
dets = [mydaq]   # just one in this case, but it could be more than one

seq_ctl = None
if args.seqctl is not None:
    if len(args.seqctl)==2:
        seq_ctl = (args.seqctl[0],int(args.seqctl[1]))
    else:
        args.c  = 0  # required for seqDone PV use
        seq_ctl = (args.seqctl[0],int(args.seqctl[1]),args.seqctl[2])

# configure BlueskyScan object with a set of motors
mydaq.configure(motors=[motor1], group_mask=args.g, events=args.c, record=args.record, detname=args.detname, scantype=args.scantype, seq_ctl=seq_ctl)

# Scan motor1 from -10 to 10, stopping
# at 15 equally-spaced points along the way and reading dets.
RE(scan(dets, motor1, -10, 10, 5))


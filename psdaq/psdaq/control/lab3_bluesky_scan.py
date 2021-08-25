# lab3_bluesky_scan.py

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
parser.add_argument('-p', type=int, choices=range(0, 8), default=1,
                    help='platform (default 1)')
parser.add_argument('-C', metavar='COLLECT_HOST', default='drp-tst-dev008',
                    help='collection host (default drp-tst-dev008)')
parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                    help='timeout msec (default 10000)')
parser.add_argument('-c', type=int, metavar='READOUT_COUNT', default=10, help='# of events to aquire at each step (default 10)')
parser.add_argument('-g', type=int, metavar='GROUP_MASK', default=2, help='bit mask of readout groups (default 2)')
parser.add_argument('--config', metavar='ALIAS', default='BEAM', help='configuration alias (default BEAM)')
parser.add_argument('--detname', default='scan', help="detector name (default 'scan')")
parser.add_argument('--scantype', default='scan', help="scan type (default 'scan')")
parser.add_argument('--record', action='store_true', help='enable recording of data')
parser.add_argument('-v', action='store_true', help='be verbose')
args = parser.parse_args()

if args.g is not None:
    if args.g < 1 or args.g > 255:
        parser.error('readout group mask (-g) must be 1-255')

if args.c < 1:
    parser.error('readout count (-c) must be >= 1')

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

# configure BlueskyScan object with a set of motors
mydaq.configure(motors=[motor1], group_mask=args.g, events=args.c, record=args.record, detname=args.detname, scantype=args.scantype)

# Scan motor1 from -10 to 10, stopping
# at 15 equally-spaced points along the way and reading dets.
RE(scan(dets, motor1, -10, 10, 15))


# bluesky_rix.py

# RIX details:
# * default platform = 2
# * one motor

from bluesky import RunEngine
from ophyd.status import Status
import sys
import logging
from psalg.utils.syslog import SysLog
import threading
import asyncio
import time

from psdaq.control.control import DaqControl
from psdaq.control.DaqScan import DaqScan
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-B', metavar='PVBASE', required=True, help='PV base')
parser.add_argument('-p', type=int, choices=range(0, 8), default=2,
                    help='platform (default 2)')
parser.add_argument('-x', metavar='XPM', type=int, required=True, help='master XPM')
parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost',
                    help='collection host (default localhost)')
parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                    help='timeout msec (default 10000)')
parser.add_argument('-c', type=int, metavar='READOUT_COUNT', default=1, help='# of events to aquire at each step (default 1)')
parser.add_argument('-g', type=int, metavar='GROUP_MASK', help='bit mask of readout groups (default 1<<plaform)')
parser.add_argument('--config', metavar='ALIAS', help='configuration alias (e.g. BEAM)')
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
instrument = control.getInstrument()
if instrument is None:
    print(f'Failed to connect with DAQ at host {args.C} platform {args.p}')
    sys.exit(1)

if args.v:
    level=logging.DEBUG
else:
    level=logging.WARNING
logger = SysLog(instrument=instrument, level=level)
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

# instantiate DaqScan object
mydaq = DaqScan(control, daqState=daqState, args=args)
dets = [mydaq]   # just one in this case, but it could be more than one

# configure MyDAQ object with a set of motors
mydaq.configure(motors=[motor1])

# Scan motor1 from -10 to 10, stopping
# at 15 equally-spaced points along the way and reading dets.
RE(scan(dets, motor1, -10, 10, 15))

mydaq.push_socket.send_string('shutdown') #shutdown the daq thread
mydaq.comm_thread.join()


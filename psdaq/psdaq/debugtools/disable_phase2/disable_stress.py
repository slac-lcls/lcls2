"""Cycle Enable/Disable many times and count Disable phase-2 failures (#237).

This is step 3 of the test in issue #237 ("cycle through many transitions,
fake motor scan").  The DAQ must already be allocated with the readout
groups under test, and the deadtime configured (steps 1 and 2); see
README.md in this directory.

Each cycle is one step of a scan:

    configured -> running (BeginStep, Enable) -> wait for StepDone
               -> starting (Disable, EndStep)

If control reports an error while going to 'starting', the cycle is counted
as a failure.  The script records the time, reads the XPM state for every
group in the partition (MsgHeader, DeadFrac/DeadTime and per-link DeadFLnk)
over PVA, then tries to recover by going to 'starting' again.  If that also
fails it stops, so the DAQ can be inspected.

Usage (FEE test stand, platform 0, XPM 10):

    python -m psdaq.debugtools.disable_phase2.disable_stress \
        -C drp-neh-ctl002 -p 0 -B DAQ:NEH -x 10 --groups 0x25 \
        --cycles 2000 --events 100 --out disable_stress_T1.jsonl
"""

import argparse
import json
import logging
import sys
import time
from threading import Event, Thread

from p4p.client.thread import Context

from psdaq.control.DaqControl import DaqControl


class Monitor(object):
    """Track control's state and error reports from monitorStatus()."""

    def __init__(self, control):
        self.control = control
        self.state = None
        self.errors = []
        self.step_done = Event()
        self.changed = Event()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            part1, part2, *_ = self.control.monitorStatus()
            if part1 is None:
                break
            if part1 == 'step':
                self.step_done.set()
            elif part1 == 'error':
                self.errors.append((time.time(), part2))
                self.changed.set()
            elif part1 not in ('warning', 'fileReport', 'progress'):
                self.state = part2
                self.changed.set()

    def wait_state(self, state, timeout):
        """True once control reaches state; False on error or timeout."""
        nerr = len(self.errors)
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.state == state:
                return True
            if len(self.errors) > nerr:
                return False
            self.changed.wait(0.5)
            self.changed.clear()
        return self.state == state


def xpm_snapshot(ctxt, base, groups):
    """Read per-group XPM state over PVA.  Missing PVs are recorded as None."""
    names = []
    for g in range(8):
        if groups & (1 << g):
            for pv in ('MsgHeader', 'DeadFrac', 'DeadTime', 'DeadFLnk', 'L0Delay'):
                names.append(f'{base}:PART:{g}:{pv}')
    values = ctxt.get(names, throw=False, timeout=2.0)
    snap = {}
    for name, val in zip(names, values):
        if isinstance(val, Exception):
            snap[name] = None
            continue
        val = val['value']
        snap[name] = val.tolist() if hasattr(val, 'tolist') else val
    return snap


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-C', required=True, help='collection (control) host')
    parser.add_argument('-p', type=int, required=True, help='platform')
    parser.add_argument('-B', required=True, help='PV base, e.g. DAQ:NEH')
    parser.add_argument('-x', type=int, required=True, help='master XPM number')
    parser.add_argument('--groups', type=lambda s: int(s, 0), required=True,
                        help='readout group mask of the partition, e.g. 0x25')
    parser.add_argument('--step_group', type=int, default=None,
                        help='readout group that counts step events (default: lowest group)')
    parser.add_argument('--cycles', type=int, default=1000, help='number of Enable/Disable cycles')
    parser.add_argument('--events', type=int, default=100, help='events per step')
    parser.add_argument('--timeout', type=float, default=60.0, help='per-transition timeout (s)')
    parser.add_argument('--out', default='disable_stress.jsonl', help='JSON-lines result file')
    parser.add_argument('-t', type=int, default=20000, help='DaqControl timeout (ms)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.v else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    step_group = args.step_group
    if step_group is None:
        step_group = (args.groups & -args.groups).bit_length() - 1
    xpm_base = f'{args.B}:XPM:{args.x}'

    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)
    mon = Monitor(control)
    ctxt = Context('pva', nt=False)

    phase1 = {'configure': {'readout_count': args.events, 'step_group': step_group,
                            'group_mask': args.groups, 'step_keys': []},
              'enable':    {'readout_count': args.events, 'step_group': step_group,
                            'group_mask': args.groups},
              'beginstep': {'step_values': {}}}

    def record(f, **kw):
        kw['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(json.dumps(kw) + '\n')
        f.flush()

    control.setState('configured', phase1)
    if not mon.wait_state('configured', args.timeout):
        sys.exit(f'could not reach configured: {mon.errors[-1:] or mon.state}')

    nfail = 0
    with open(args.out, 'a') as f:
        record(f, event='start', args=vars(args))
        for cycle in range(args.cycles):
            mon.step_done.clear()
            control.setState('running', phase1)
            if not mon.wait_state('running', args.timeout):
                record(f, event='enable_failed', cycle=cycle, errors=mon.errors[-5:])
                logging.error(f'cycle {cycle}: Enable failed, stopping')
                break
            if not mon.step_done.wait(args.timeout):
                logging.warning(f'cycle {cycle}: no StepDone, disabling anyway')

            t_dis = time.time()
            control.setState('starting')
            if mon.wait_state('starting', args.timeout):
                continue

            nfail += 1
            snap = xpm_snapshot(ctxt, xpm_base, args.groups)
            record(f, event='disable_failed', cycle=cycle, t_disable=t_dis,
                   errors=[e for e in mon.errors if e[0] >= t_dis], xpm=snap)
            logging.error(f'cycle {cycle}: Disable failed ({nfail} so far); retrying')

            control.setState('starting')
            if not mon.wait_state('starting', args.timeout):
                record(f, event='retry_failed', cycle=cycle, errors=mon.errors[-5:])
                logging.error(f'cycle {cycle}: retry failed, stopping for inspection')
                break
            record(f, event='retry_ok', cycle=cycle)

        record(f, event='end', cycles=cycle + 1, failures=nfail)

    logging.info(f'{cycle + 1} cycles, {nfail} Disable failures; results in {args.out}')
    control.setState('configured')
    return 0 if nfail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

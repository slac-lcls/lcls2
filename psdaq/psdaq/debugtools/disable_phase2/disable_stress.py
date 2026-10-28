#!/usr/bin/env python3
"""Timed Enable/Disable cycles; stop on first error, with no recovery writes."""
import argparse
import json
import random
import sys
import time
from pathlib import Path


class Failure(RuntimeError):
    pass


def active_groups(platform):
    groups = set()
    for level in ('drp', 'tpr'):
        for node in platform.get(level, {}).values():
            if node.get('active'):
                groups.add(int(node['det_info']['readout']))
    return groups


class Driver:
    def __init__(self, control, timeout, record):
        self.control, self.timeout, self.record = control, timeout, record

    def messages(self):
        while self.control.front_sub.poll(0):
            msg = self.control.front_sub.recv_json()
            key, body = msg['header']['key'], msg.get('body', {})
            if key in ('error', 'warning'):
                self.record(event=key, body=body)
            if key == 'error':
                raise Failure(body.get('err_info', str(body)))
            yield key, body

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            list(self.messages())
            time.sleep(min(0.05, max(0, deadline-time.monotonic())))
        list(self.messages())

    def change(self, state, phase1=None):
        list(self.messages())
        err = self.control.setState(state, phase1 or {})
        if err:
            raise Failure(err)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            reached = False
            for key, body in self.messages():
                if key == 'status' and body.get('state') == state:
                    reached = True
            if reached:
                return
            self.control.front_sub.poll(100)
        raise Failure(f'timeout waiting for {state}')


def snapshot(context, groups, xpm_base):
    names = [f'{xpm_base}:PART:{g}:{field}' for g in sorted(groups)
             for field in ('NumL0Acc', 'DeadFrac', 'DeadTime', 'DeadFLnk', 'L0Delay')]
    try:
        values = context.get(names, timeout=3, throw=False)
        return {name: (None if isinstance(value, Exception) else
                       value.tolist() if hasattr(value, 'tolist') else value)
                for name, value in zip(names, values)}
    except Exception as exc:
        return {'snapshot_error': str(exc)}


def require_triggers(before, after, groups, xpm_base):
    for group in groups:
        pv = f'{xpm_base}:PART:{group}:NumL0Acc'
        if before.get(pv) is None or after.get(pv) is None:
            raise Failure(f'cannot verify trigger counts for group {group}')
        if after[pv] <= before[pv]:
            raise Failure(f'no verified trigger-count increase in group {group}')


def run(control, context, args, record):
    attempted = delivered = completed = 0
    stage = 'preflight'
    groups = set()
    driver = Driver(control, args.timeout, record)
    try:
        status = control.getStatus()
        initial_state = 'configured' if args.use_configured else 'connected'
        if status[1] != initial_state:
            raise Failure(f'Expected {initial_state} in the GUI, got {status[1]}')
        if status[3] is not False:
            raise Failure('Recording must be OFF for this initial reproducer')
        groups = active_groups(status[4])
        expected = {g for g in range(8) if args.groups & (1 << g)}
        if groups != expected:
            raise Failure(f'active groups {sorted(groups)} differ from expected {sorted(expected)}')
        record(event='start', args=vars(args), groups=sorted(groups), experiment=status[6])
        # Allow the subscription to connect before making state transitions.
        driver.wait(0.5)
        phase1 = {'configure': {'readout_count': 0, 'group_mask': args.groups,
                               'step_group': min(groups), 'step_keys': []},
                  'enable': {'readout_count': 0}, 'beginstep': {'step_values': {}}}
        stage = 'configure'
        if not args.use_configured:
            driver.change('configured', phase1)
        driver.wait(1.2)
        for cycle in range(1, args.cycles + 1):
            before = snapshot(context, groups, args.xpm_base)
            stage = 'enable'
            driver.change('running', phase1)
            stage = 'acquire'
            dwell = random.uniform(args.dwell, args.dwell + args.jitter)
            driver.wait(dwell)
            stage = 'disable'
            attempted += 1
            record(event='disable_attempt', cycle=cycle, attempted=attempted, dwell=dwell)
            # Running -> Paused is Disable only. EndStep is tested separately.
            driver.change('paused')
            delivered += 1
            stage = 'verify_triggers'
            driver.wait(1.2)  # allow the published statistics interval to catch up
            after = snapshot(context, groups, args.xpm_base)
            record(event='trigger_check', cycle=cycle, before=before, after=after)
            require_triggers(before, after, groups, args.xpm_base)
            stage = 'endstep'
            driver.change('starting')
            completed += 1
            record(event='cycle_ok', cycle=cycle, before=before, after=after)
            if cycle == 1 or cycle % 25 == 0:
                print(f'{completed}/{args.cycles} valid cycles, no transition failures', flush=True)
        stage = 'endrun'
        driver.change('configured')
        record(event='end', outcome='complete', attempted=attempted,
               delivered=delivered, completed=completed)
        print(f'COMPLETE: {completed} valid cycles. Left CONFIGURED; no retry was used.')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        record(event='end', outcome='interrupted' if interrupted else 'failed',
               stage=stage, error=str(exc), attempted=attempted,
               delivered=delivered, completed=completed, snapshot=snapshot(context, groups, args.xpm_base))
        print(f'STOPPED at {stage}: {exc}. {attempted} Disable attempts, {completed} valid cycles.', file=sys.stderr)
        print('No retry or cleanup transition sent. Inspect the DAQ and process logs before recovery.', file=sys.stderr)
        return 130 if interrupted else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', '-C', required=True, help='control host')
    parser.add_argument('--platform', '-p', type=int, required=True)
    parser.add_argument('--groups', type=lambda x: int(x, 0), required=True)
    parser.add_argument('--xpm-base', required=True, help='PV prefix, e.g. DAQ:NEH:XPM:2')
    parser.add_argument('--cycles', type=int, default=20)
    parser.add_argument('--dwell', type=float, default=2.0, help='minimum acquisition seconds per step')
    parser.add_argument('--jitter', type=float, default=1.0, help='additional random acquisition seconds')
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--use-configured', action='store_true',
                        help='start from Configured; preserve timing-panel settings applied after Configure')
    parser.add_argument('--label', choices=('baseline', 'candidate'), required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.cycles < 1 or args.dwell < 1 or args.jitter < 0 or args.timeout < 60 or not 0 < args.groups < 256:
        parser.error('require cycles >= 1, dwell >= 1, jitter >= 0, timeout >= 60, and an 8-bit nonzero group mask')
    # Import only after parsing so --help can be used without DAQ libraries.
    from psdaq.control.DaqControl import DaqControl
    from p4p.client.thread import Context
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'x') as stream:
        def record(**row):
            row['time'] = time.time()
            stream.write(json.dumps(row) + '\n')
            stream.flush()
        control = DaqControl(host=args.host, platform=args.platform, timeout=int(args.timeout * 1000))
        context = Context('pva')
        try:
            return run(control, context, args, record)
        finally:
            context.close()


if __name__ == '__main__':
    sys.exit(main())

"""Hardware-free checks for failure accounting and preservation of failed state."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from psdaq.debugtools.disable_phase2 import disable_stress as reproduce


class Socket:
    def __init__(self):
        self.queue = []

    def poll(self, timeout):
        return bool(self.queue)

    def recv_json(self):
        return self.queue.pop(0)


class Control:
    def __init__(self, fail=None, recording=False, initial_state='connected'):
        self.commands = []
        self.front_sub = Socket()
        self.fail = fail
        self.recording = recording
        self.initial_state = initial_state

    def getStatus(self):
        nodes = {str(g): {'active': 1, 'det_info': {'readout': g}} for g in (0, 2, 5)}
        return ('connect', self.initial_state, 'BEAM', self.recording, {'drp': nodes}, False, 'test', 0, 0)

    def setState(self, state, info):
        self.commands.append(state)
        if state == self.fail:
            self.front_sub.queue.append({'header': {'key': 'error'}, 'body': {'err_info': 'injected failure'}})
        else:
            self.front_sub.queue.append({'header': {'key': 'status'}, 'body': {'state': state}})


class HarnessTests(unittest.TestCase):
    def exercise(self, fail=None, recording=False, counts=True, use_configured=False):
        control = Control(fail, recording, 'configured' if use_configured else 'connected')
        rows = []
        args = SimpleNamespace(timeout=120, groups=0x25, cycles=2, dwell=2, jitter=0, use_configured=use_configured, xpm_base='DAQ:NEH:XPM:10')
        count = 0

        def snapshot(context, groups, xpm_base):
            nonlocal count
            count += 100 if counts else 0
            return {f'{xpm_base}:PART:{g}:NumL0Acc': count for g in groups}

        with patch.object(reproduce.Driver, 'wait', return_value=None), patch.object(reproduce, 'snapshot', side_effect=snapshot):
            rc = reproduce.run(control, None, args, lambda **row: rows.append(row))
        return control.commands, rows[-1], rc

    def test_success_counts_only_completed_cycles(self):
        commands, end, rc = self.exercise()
        self.assertEqual(rc, 0)
        self.assertEqual((end['attempted'], end['delivered'], end['completed']), (2, 2, 2))
        self.assertEqual(commands[-1], 'configured')

    def test_enable_failure_is_failure_with_zero_disables(self):
        commands, end, rc = self.exercise('running')
        self.assertEqual(rc, 1)
        self.assertEqual((end['attempted'], end['completed']), (0, 0))
        self.assertEqual(commands, ['configured', 'running'])

    def test_disable_failure_does_not_retry_or_cleanup(self):
        commands, end, rc = self.exercise('paused')
        self.assertEqual(rc, 1)
        self.assertEqual((end['attempted'], end['delivered'], end['completed']), (1, 0, 0))
        self.assertEqual(commands, ['configured', 'running', 'paused'])
        self.assertEqual(end['stage'], 'disable')

    def test_endstep_failure_is_distinct(self):
        commands, end, rc = self.exercise('starting')
        self.assertEqual(rc, 1)
        self.assertEqual((end['attempted'], end['delivered'], end['completed']), (1, 1, 0))
        self.assertEqual(end['stage'], 'endstep')
        self.assertEqual(commands[-1], 'starting')

    def test_no_triggers_cannot_pass(self):
        commands, end, rc = self.exercise(counts=False)
        self.assertEqual(rc, 1)
        self.assertEqual(end['completed'], 0)
        self.assertEqual(end['stage'], 'verify_triggers')
        self.assertEqual(commands[-1], 'paused')

    def test_recording_preflight_has_no_state_changes(self):
        commands, end, rc = self.exercise(recording=True)
        self.assertEqual(rc, 1)
        self.assertEqual(commands, [])

    def test_configured_entry_does_not_reconfigure(self):
        commands, end, rc = self.exercise(use_configured=True)
        self.assertEqual(rc, 0)
        self.assertEqual(commands[0], 'running')
        self.assertEqual(commands.count('configured'), 1)  # final EndRun only

    def test_synchronous_command_error_stops_immediately(self):
        control = Control()
        control.setState = Mock(return_value='request rejected')
        driver = reproduce.Driver(control, 120, lambda **row: None)
        with self.assertRaisesRegex(reproduce.Failure, 'request rejected'):
            driver.change('paused')
        control.setState.assert_called_once_with('paused', {})

    def test_error_wins_over_status_in_same_batch(self):
        control = Control()

        def set_state(state, info):
            control.front_sub.queue.extend([
                {'header': {'key': 'status'}, 'body': {'state': state}},
                {'header': {'key': 'error'}, 'body': {'err_info': 'phase failed'}}])

        control.setState = set_state
        driver = reproduce.Driver(control, 120, lambda **row: None)
        with self.assertRaisesRegex(reproduce.Failure, 'phase failed'):
            driver.change('paused')

    def test_snapshot_uses_requested_xpm_and_preserves_read_errors(self):
        context = Mock()
        context.get.return_value = [123.0, 0.0, 0.5, [0.0], RuntimeError('unavailable')]
        result = reproduce.snapshot(context, {2}, 'DAQ:TEST:XPM:10')
        names = context.get.call_args[0][0]
        self.assertTrue(all(n.startswith('DAQ:TEST:XPM:10:PART:2:') for n in names))
        self.assertEqual(result[names[0]], 123.0)
        self.assertEqual(result[names[3]], [0.0])
        self.assertIsNone(result[names[4]])

    def test_one_missing_group_cannot_pass(self):
        prefix = 'DAQ:TEST:XPM:10'
        before = {f'{prefix}:PART:{g}:NumL0Acc': 0 for g in (0, 2, 5)}
        after = {f'{prefix}:PART:{g}:NumL0Acc': 100 for g in (0, 2, 5)}
        after[f'{prefix}:PART:2:NumL0Acc'] = 0
        with self.assertRaisesRegex(reproduce.Failure, 'group 2'):
            reproduce.require_triggers(before, after, {0, 2, 5}, prefix)


if __name__ == '__main__':
    unittest.main()

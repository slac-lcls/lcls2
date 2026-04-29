#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from psdaq.control.DaqControl import DaqControl
from psdaq.control.timed_run import deduce_platform3
from psdaq.debugtools.epixquad1kfps.pattern_loader import control_file_path


def _load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Run one DAQ run per ePix quad debug pattern'
    )
    parser.add_argument('--sequence', required=True,
                        help='Pattern-sequence JSON file')
    parser.add_argument('--daq-config', default=None,
                        help='DAQ Python config file used to resolve platform/collect host; required unless --dry-run is used')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Run duration per pattern in seconds (default 2.0)')
    parser.add_argument('--config-alias', default=None,
                        help='DAQ config alias to set before running (default uses sequence defaults)')
    parser.add_argument('--record', action='store_true',
                        help='Enable recording')
    parser.add_argument('--outdir', default=None,
                        help='Optional directory for materialized pattern .npy/.json sidecars')
    parser.add_argument('--patterns', default=None,
                        help='Comma-separated subset of pattern_index values to run')
    parser.add_argument('--steps', default=None,
                        help='Deprecated alias for --patterns')
    parser.add_argument('--timeout', type=float, default=20.0,
                        help='Timeout in seconds for DAQ state changes (default 20)')
    parser.add_argument('--control-file', default=None,
                        help='Optional override path for the shared pattern control JSON')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print planned runs and exit without touching DAQ')
    parser.add_argument('-v', action='store_true',
                        help='Verbose logging')
    return parser.parse_args()


def _sequence_patterns(sequence_spec):
    return sequence_spec.get('patterns', sequence_spec.get('steps', []))


def _pattern_index(pattern, default_index):
    if 'pattern_index' in pattern:
        return int(pattern['pattern_index'])
    if 'step_index' in pattern:
        return int(pattern['step_index'])
    return int(default_index)


def _selected_patterns(sequence_spec, patterns_arg):
    patterns = sorted(
        _sequence_patterns(sequence_spec),
        key=lambda item_with_index: _pattern_index(item_with_index, 0),
    )
    if not patterns_arg:
        return patterns
    selected = {int(s.strip()) for s in patterns_arg.split(',') if s.strip()}
    return [
        pattern for i, pattern in enumerate(patterns)
        if _pattern_index(pattern, i) in selected
    ]


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _write_control_file(path, sequence_file, pattern, pattern_outdir, expires_at):
    payload = {
        'enabled': True,
        'sequence_file': str(Path(sequence_file).resolve()),
        'pattern_index': int(_pattern_index(pattern, 0)),
        'expires_at': float(expires_at),
    }
    if pattern_outdir:
        payload['pattern_outdir'] = str(Path(pattern_outdir).resolve())
    _write_json_atomic(path, payload)


def _disable_control_file(path):
    _write_json_atomic(path, {
        'enabled': False,
        'updated_at': time.time(),
    })


def _wait_for_state(control, target_state, timeout_sec):
    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        last = control.getStatus()
        if last[1] == target_state:
            return last
        time.sleep(0.25)
    raise TimeoutError(f'timed out waiting for DAQ state {target_state}; last status={last}')


def _set_state_and_wait(control, target_state, timeout_sec):
    err = control.setState(target_state)
    if err is not None:
        raise RuntimeError(f'setState({target_state}) failed: {err}')
    return _wait_for_state(control, target_state, timeout_sec)


def _log_step_line(log_path, payload):
    if log_path is None:
        return
    with Path(log_path).open('a') as f:
        f.write(json.dumps(payload, sort_keys=True) + '\n')


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.v else logging.INFO,
                        format='[%(levelname).1s] %(message)s')

    if not args.dry_run and not args.daq_config:
        raise RuntimeError('--daq-config is required unless --dry-run is used')

    sequence_path = Path(args.sequence).expanduser().resolve()
    sequence_spec = _load_json(sequence_path)
    selected_arg = args.patterns if args.patterns is not None else args.steps
    patterns = _selected_patterns(sequence_spec, selected_arg)
    if not patterns:
        raise RuntimeError('no patterns selected')

    control_path = Path(args.control_file or control_file_path()).expanduser().resolve()
    pattern_outdir = None if args.outdir is None else Path(args.outdir).expanduser().resolve()
    sequence_defaults = sequence_spec.get('defaults', {})
    config_alias = args.config_alias if args.config_alias is not None else sequence_defaults.get('config_alias')
    record = bool(args.record or sequence_defaults.get('record', False))

    print(f'sequence: {sequence_spec.get("sequence_name", "unnamed_sequence")}')
    print(f'patterns: [{", ".join(str(_pattern_index(pattern, i)) for i, pattern in enumerate(patterns))}]')
    print(f'control_file: {control_path}')
    if pattern_outdir:
        print(f'pattern_outdir: {pattern_outdir}')

    for i, pattern in enumerate(patterns):
        print(
            f'  pattern {_pattern_index(pattern, i):02d}: '
            f'{pattern.get("label", "")} '
            f'groups={pattern.get("marker_groups", [])} '
            f'group_index={pattern.get("group_index", 0)}'
        )

    if args.dry_run:
        return

    platform, collect_host = deduce_platform3(args.daq_config)
    if platform == -1:
        raise RuntimeError('failed to resolve platform/collect_host from DAQ config file')

    control = DaqControl(host=collect_host, platform=platform, timeout=int(args.timeout * 1000))
    instrument = control.getInstrument()
    if instrument is None:
        raise RuntimeError('failed to read instrument name from DAQ control')

    if config_alias:
        err = control.setConfig(config_alias)
        if err is not None:
            raise RuntimeError(f'setConfig({config_alias}) failed: {err}')

    err = control.setRecord(record)
    if err is not None:
        raise RuntimeError(f'setRecord({record}) failed: {err}')

    log_path = None
    if pattern_outdir is not None:
        pattern_outdir.mkdir(parents=True, exist_ok=True)
        log_path = pattern_outdir / f'{sequence_spec.get("sequence_name", "sequence")}_run_log.jsonl'

    try:
        for i, pattern in enumerate(patterns):
            pattern_index = _pattern_index(pattern, i)
            expires_at = time.time() + max(args.duration, 1.0) + args.timeout + 30.0
            _write_control_file(control_path, sequence_path, pattern, pattern_outdir, expires_at)

            logging.info('starting pattern %02d label=%s groups=%s group_index=%s',
                         pattern_index, pattern.get('label', ''), pattern.get('marker_groups', []),
                         pattern.get('group_index', 0))

            _set_state_and_wait(control, 'connected', args.timeout)
            status_before = control.getStatus()
            _set_state_and_wait(control, 'running', args.timeout)
            status_running = control.getStatus()
            run_number = status_running[7]
            logging.info('run started: run_number=%s pattern=%02d label=%s',
                         str(run_number), pattern_index, pattern.get('label', ''))

            time.sleep(args.duration)

            status_after = _set_state_and_wait(control, 'connected', args.timeout)
            last_run_number = status_after[8]
            logging.info('run ended: last_run_number=%s pattern=%02d label=%s',
                         str(last_run_number), pattern_index, pattern.get('label', ''))

            _log_step_line(log_path, {
                'timestamp': time.time(),
                'sequence_name': sequence_spec.get('sequence_name', 'unnamed_sequence'),
                'pattern_index': pattern_index,
                'pattern_label': pattern.get('label', ''),
                'marker_groups': pattern.get('marker_groups', []),
                'group_index': pattern.get('group_index', 0),
                'test_file': pattern.get('test_file', ''),
                'duration_s': args.duration,
                'record': record,
                'config_alias': config_alias,
                'run_number_running': run_number,
                'last_run_number_connected': last_run_number,
                'status_before': {'transition': status_before[0], 'state': status_before[1]},
                'status_running': {'transition': status_running[0], 'state': status_running[1]},
                'status_after': {'transition': status_after[0], 'state': status_after[1]},
            })
    finally:
        _disable_control_file(control_path)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('interrupted')

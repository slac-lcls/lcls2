"""Balanced, separate-process comparison of frozen event/batch runtimes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event-runtime', type=Path, required=True)
    p.add_argument('--batch-runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--micro-submissions', type=int, default=200)
    p.add_argument('--jungfrau-submissions', type=int, default=50)
    a = p.parse_args()
    if a.rounds < 2 or a.rounds % 2:
        p.error('use an even number of balanced rounds >= 2')
    a.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('callback_cost.py')
    cases = []
    start = time.monotonic()
    for round_id in range(1, a.rounds+1):
        profiles = ('micro', 'jungfrau') if round_id % 2 else ('jungfrau', 'micro')
        order = ('event', 'batch') if round_id % 2 else ('batch', 'event')
        for profile in profiles:
            for dispatch in order:
                runtime = a.event_runtime if dispatch == 'event' else a.batch_runtime
                env = dict(os.environ, PYTHONPATH=str(runtime)+os.pathsep+os.environ.get('PYTHONPATH', ''))
                name = f'r{round_id}-{profile}-{dispatch}'
                output = a.output/(name+'.json')
                command = [sys.executable, str(script), '--output', str(output),
                           '--dispatch', dispatch, '--profile', profile, '--repetitions', '1',
                           '--submissions', str(a.micro_submissions if profile == 'micro' else a.jungfrau_submissions)]
                if round_id % 2 == 0: command.append('--reverse')
                print('CALLBACK_CASE_START', name, flush=True)
                case_start = time.monotonic()
                with (a.output/(name+'.log')).open('w') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                data = json.loads(output.read_text())
                assert data['complete'] and data['dispatch'] == dispatch and data['profile'] == profile
                assert Path(data['psana']).resolve().is_relative_to(runtime.resolve())
                assert len(data['preflights']) == len(data['samples']) == 36
                cases.append(dict(round=round_id, dispatch=dispatch, profile=profile,
                                  seconds=time.monotonic()-case_start, output=str(output), data=data))
                (a.output/'progress.json').write_text(json.dumps([
                    {k: v for k, v in c.items() if k != 'data'} for c in cases], indent=2)+'\n')
                print('CALLBACK_CASE_COMPLETE', name, round(time.monotonic()-case_start, 2), flush=True)
    summary = []
    for profile in ('micro', 'jungfrau'):
        for size in (1, 3, 20):
            for depth in (1, 2):
                for mode in ('none', 'empty', 'scratch', 'publish', 'scratch_prealloc', 'publish_prealloc'):
                    pairs = []
                    for round_id in range(1, a.rounds+1):
                        pair = {}
                        for dispatch in ('event', 'batch'):
                            case = next(c for c in cases if (c['round'], c['profile'], c['dispatch']) == (round_id, profile, dispatch))
                            pair[dispatch] = next(r for r in case['data']['samples'] if (r['batch_size'], r['depth'], r['mode']) == (size, depth, mode))
                        pairs.append(pair)
                    row = dict(profile=profile, batch_size=size, depth=depth, mode=mode)
                    for phase in ('submit', 'retire', 'loop'):
                        for unit in ('subbatch', 'event'):
                            key = phase+'_us_per_'+unit
                            row[key] = dict(event_median=statistics.median(p['event'][key] for p in pairs),
                                batch_median=statistics.median(p['batch'][key] for p in pairs),
                                paired_delta_median=statistics.median(p['batch'][key]-p['event'][key] for p in pairs),
                                paired_deltas=[p['batch'][key]-p['event'][key] for p in pairs],
                                paired_percent=[100*(p['batch'][key]/p['event'][key]-1) for p in pairs])
                    summary.append(row)
    (a.output/'summary.json').write_text(json.dumps(dict(complete=True, rounds=a.rounds,
        wall_seconds=time.monotonic()-start, event_runtime=str(a.event_runtime),
        batch_runtime=str(a.batch_runtime), summary=summary), indent=2)+'\n')
    print('CALLBACK_COMPARISON_COMPLETE', round(time.monotonic()-start, 2), flush=True)


if __name__ == '__main__':
    main()

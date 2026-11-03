"""Report, per readout group, which DRPs received each failed Disable.

For every ``disable phase2 failed`` in a launch's control.log, find the
Disable start (the preceding ``slowupdate transitions DISABLED`` line), then
for each DRP log in the same launch check whether ``PGPReader saw Disable``
follows its matching ``received key = disable``.  DRPs are grouped by the
``Readout group receipient`` they print at Configure.

A group whose DRPs all miss the Disable while another group's DRPs all get
it is the signature of issue #237.

Usage:
    python -m psdaq.debugtools.disable_phase2.check_disable_groups \
        --logdir /sdf/home/t/tmoopr/daq/logs/2026 10/25_13:11:29 10/26_06:31:55
"""

import argparse
import datetime
import glob
import os
import re
import subprocess
import sys

NOT_DRP = ('teb', 'meb', 'control', 'ami', 'acr', 'prom', 'group', 'daqstat',
           'pyxpm', 'pvrt', 'ioc', 'hsdpvs', 'xpmpva')
RE_KEY = re.compile(r'received key = disable @ (.*)$')
RE_PGP = re.compile(r'PGPReader +saw +Disable')
RE_RCPT = re.compile(r'Readout group receipient: +(\S+)')


def read_lines(path):
    if path.endswith('.zst'):
        out = subprocess.run(['zstdcat', path], capture_output=True, text=True,
                             errors='replace').stdout
        return out.splitlines()
    with open(path, errors='replace') as f:
        return f.read().splitlines()


def failed_disables(launch):
    """Return the start time of each failed Disable in this launch."""
    paths = [p for p in glob.glob(f'{launch}_*control.log*') if 'control_gui' not in p]
    if not paths:
        return []
    lines = read_lines(paths[0])
    starts = []
    for i, line in enumerate(lines):
        if 'disable phase2 failed' not in line:
            continue
        t = datetime.datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
        for j in range(i - 1, max(0, i - 50), -1):
            if 'slowupdate transitions DISABLED' in lines[j]:
                t = datetime.datetime.strptime(lines[j][:19], '%Y-%m-%d %H:%M:%S')
                break
        starts.append(t)
    return starts


def drp_result(lines, tdis, window=3):
    """(group, got) for one DRP log, or None if it has no matching Disable."""
    group = '?'
    for i, line in enumerate(lines):
        m = RE_RCPT.search(line)
        if m:
            group = m.group(1)
        m = RE_KEY.search(line)
        if not m:
            continue
        t = datetime.datetime.strptime(m.group(1).strip(), '%a %b %d %H:%M:%S %Y')
        if abs((t - tdis).total_seconds()) > window:
            continue
        for k in range(i + 1, min(len(lines), i + 40)):
            if 'received key' in lines[k]:
                break
            if RE_PGP.search(lines[k]):
                return group, True
        return group, False
    return None


def check_launch(launch):
    rows = []
    drp_logs = sorted(glob.glob(f'{launch}_*.log') + glob.glob(f'{launch}_*.log.zst'))
    drp_logs = [p for p in drp_logs if not any(s in os.path.basename(p) for s in NOT_DRP)]
    cache = {p: read_lines(p) for p in drp_logs}
    for tdis in failed_disables(launch):
        groups = {}
        for path in drp_logs:
            res = drp_result(cache[path], tdis)
            if res is None:
                continue
            name = re.sub(r'\.log.*', '', os.path.basename(path).split(':')[-1])
            got = groups.setdefault(res[0], ([], []))
            got[0 if res[1] else 1].append(name)
        rows.append((tdis, groups))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--logdir', required=True, help='e.g. /sdf/home/t/tmoopr/daq/logs/2026')
    parser.add_argument('launches', nargs='+', help='launch prefixes, e.g. 10/25_13:11:29')
    args = parser.parse_args()

    os.chdir(args.logdir)
    for launch in args.launches:
        rows = check_launch(launch)
        if not rows:
            print(f'== {launch}: no "disable phase2 failed"')
        for tdis, groups in rows:
            print(f'== {launch}  Disable @ {tdis.time()}')
            for g in sorted(groups):
                ok, bad = groups[g]
                state = 'ALL GOT' if not bad else ('NONE GOT' if not ok else 'MIXED')
                print(f'    group {g:5} {state:8}  got: {" ".join(ok) or "-"}  | missed: {" ".join(bad) or "-"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

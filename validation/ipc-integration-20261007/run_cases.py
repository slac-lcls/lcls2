"""Drive the psana2-ipc integration cases, one MPI job per case.

Replaces a shell loop of back-to-back `mpirun` calls, which hung: consecutive
launches inside one allocation share PRRTE/session state, and a case that
aborts can leave the next one unable to start. Each case here gets its own
session directory and its own log file, is hard-bounded by a timeout, and its
outcome is parsed from the log rather than inferred from an exit status --
`mpirun` returns 0 even when ranks report failures.

Usage, from the repository root inside a GPU allocation::

    python validation/ipc-integration-20261007/run_cases.py
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
TESTS = REPO / 'psana/psana/tests/gpu/integration'
LOGS = HERE / f'cases-{time.strftime("%Y%m%d-%H%M%S")}'

DISCOVERY = TESTS / 'mpi_placement_discovery.py'
SHARING = TESTS / 'mpi_shared_constants.py'
FAILURES = TESTS / 'mpi_failure_paths.py'

# (tag, ranks, script, extra argv, one_gpu, expect)
#
# one_gpu restricts the mask so every worker is a peer, which is what sharing
# needs; discovery deliberately spans whatever the launcher provided.
#
# expect='clean'  every rank reports PASS and no failures
# expect='abort'  the job must DIE, promptly -- nonzero exit, the expected
#                 message in the log, and well inside the timeout. The timing
#                 clause is the point: a correct abort and a regressed
#                 deadlock produce near-identical logs, and without it a hang
#                 would pass.
CASES = (
    ('discovery-eb1',        9, DISCOVERY,
     ['--n-eb', '1', '--expect-hosts', '1', '--expect-devices', '2'], False, 'clean'),
    ('discovery-eb2',        9, DISCOVERY,
     ['--n-eb', '2', '--expect-hosts', '1', '--expect-devices', '2'], False, 'clean'),
    ('discovery-eb2-contig', 9, DISCOVERY,
     ['--n-eb', '2', '--contiguous', '--expect-hosts', '1'], False, 'clean'),
    ('discovery-eb3',        9, DISCOVERY,
     ['--n-eb', '3', '--expect-hosts', '1'], False, 'clean'),
    ('discovery-uneven',     8, DISCOVERY,
     ['--n-eb', '2', '--contiguous', '--expect-hosts', '1'], False, 'clean'),
    ('sharing-2-peers',      3, SHARING,   [], True, 'clean'),
    ('sharing-4-peers',      5, SHARING,   [], True, 'clean'),
    # Failure paths. The verify-pin case is the only test of the abort
    # reaching EB and smd0, which the device-group allreduce cannot touch.
    ('fail-verify-pin',      5, FAILURES, ['--inject', 'verify-pin'], False, 'abort'),
    ('fail-follower-import', 4, FAILURES, ['--inject', 'follower-import'], True, 'clean'),
)


def run_case(tag, ranks, script, extra, one_gpu, expect='clean', timeout=180):
    """Launch one MPI job in its own session and return (ok, detail)."""
    env = os.environ.copy()
    # A private PRRTE session per case: without this, a case that aborts can
    # leave state that prevents the next `mpirun` from starting.
    session = LOGS / f'session-{tag}'
    session.mkdir(parents=True, exist_ok=True)
    env['TMPDIR'] = str(session)
    env['OMPI_MCA_orte_tmpdir_base'] = str(session)
    env['PMIX_MCA_ptl_tcp_if_include'] = env.get('PMIX_MCA_ptl_tcp_if_include', 'lo')
    if one_gpu:
        env['CUDA_VISIBLE_DEVICES'] = '0'

    command = ['mpirun', '-n', str(ranks), 'python', '-u', str(script), *extra]
    log = LOGS / f'{tag}.log'
    print(f'CASE_BEGIN {tag} ranks={ranks} expect={expect} '
          f'{" ".join(extra) or "-"}', flush=True)
    status = None
    began = time.monotonic()
    timed_out = False
    with log.open('w') as handle:
        try:
            status = subprocess.run(command, cwd=REPO, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT,
                                    timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    elapsed = round(time.monotonic() - began, 1)

    text = log.read_text()
    if expect == 'abort':
        ok, detail = judge_abort(text, status, elapsed, timeout, timed_out)
    elif timed_out:
        ok, detail = False, {'error': f'timed out after {timeout}s'}
    else:
        ok, detail = parse(text)
    detail['exit'] = status
    detail['elapsed'] = elapsed
    # mpirun exits 0 even when ranks report failures, so the log decides.
    print(f'CASE_END {tag} ok={ok} exit={status} elapsed={elapsed}s '
          f'{json.dumps(detail, sort_keys=True)}', flush=True)
    return ok, detail


def judge_abort(text, status, elapsed, timeout, timed_out):
    """An abort case passes only if the job died, and died quickly.

    Three conditions, all required:

    * it did not time out -- a deadlock is the regression being guarded
      against, and it leaves a log almost identical to a clean abort;
    * the exit status is nonzero, so the failure propagated to the launcher;
    * it finished well inside the timeout, so "aborted" is distinguishable
      from "nearly hung".
    """
    detail = {'expected': 'abort', 'timed_out': timed_out}
    budget = 0.5 * timeout
    detail['time_budget'] = budget
    aborted = ('MPI_ABORT' in text or 'Abort' in text
               or 'GpuPlacementError' in text)
    detail['abort_evidence'] = aborted
    detail['survived'] = 'SURVIVED' in text
    if detail['survived']:
        detail['error'] = 'a rank continued past the injected failure'
        return False, detail
    if timed_out:
        detail['error'] = (f'hung for the full {timeout}s instead of aborting; '
                           'ranks are probably in different collectives')
        return False, detail
    if status in (0, None):
        detail['error'] = f'exited {status}; the failure did not propagate'
        return False, detail
    if elapsed > budget:
        detail['error'] = (f'took {elapsed}s of a {timeout}s timeout; too slow '
                           'to distinguish an abort from a near-hang')
        return False, detail
    if not aborted:
        detail['error'] = 'no abort or placement error in the log'
        return False, detail
    return True, detail


def parse(text):
    """Outcome from the rank and summary records a case emits."""
    detail = {}
    ranks_pass = text.count('"PASS": true')
    ranks_fail = text.count('"PASS": false')
    detail['ranks_pass'] = ranks_pass
    detail['ranks_fail'] = ranks_fail

    failures = []
    for block in re.findall(r'"failures": \[(.*?)\]', text, re.S):
        failures.extend(f.strip().strip('"') for f in block.split('\n')
                        if f.strip() and f.strip() != ',')
    detail['failures'] = sorted(set(failures))[:5]

    devices = []
    for block in re.findall(r'DISCOVERY_DEVICE (\{.*?\n\})', text, re.S):
        record = json.loads(block)
        devices.append({'peers': record['design_peers'],
                        'agg': record['aggregate_claim'],
                        'pass': record['all_pass']})
    if devices:
        detail['devices'] = devices

    for block in re.findall(r'SHARED_SUMMARY (\{.*?\n\})', text, re.S):
        record = json.loads(block)
        detail['sharing'] = {'peers': record['peers'],
                             'all_pass': record['all_pass'],
                             'charged': record['charged_by_rank'],
                             'imported': record['imported_by_rank']}

    # PSANA_GPU_CHECK_COLLECTIVES=1 makes each peer compare its collective
    # sequence with the others. A divergence is reported rather than raised,
    # so the launcher has to fail the case on it or the check is advisory.
    diverged = text.count('collective order diverged')
    detail['collective_order_diverged'] = diverged
    if diverged:
        return False, {**detail,
                       'error': f'{diverged} rank(s) reported a diverging '
                                'collective order'}

    if ranks_pass == 0 and ranks_fail == 0:
        return False, {**detail, 'error': 'no rank records emitted'}
    ok = ranks_fail == 0 and not detail['failures']
    if devices:
        ok = ok and all(d['pass'] for d in devices)
    if 'sharing' in detail:
        ok = ok and detail['sharing']['all_pass']
    return ok, detail


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f'LOG_DIRECTORY {LOGS}', flush=True)
    results = {}
    for tag, ranks, script, extra, one_gpu, expect in CASES:
        ok, _ = run_case(tag, ranks, script, extra, one_gpu, expect=expect)
        results[tag] = ok

    print('\n=============================== VERDICT ===============================',
          flush=True)
    for tag, ok in results.items():
        print(f'  {"PASS" if ok else "FAIL"}  {tag}', flush=True)
    overall = all(results.values())
    print(f'  OVERALL: {"PASS" if overall else "FAIL"}', flush=True)
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())

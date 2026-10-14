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
REALPATH = TESTS / 'mpi_real_path_placement.py'
SHAREDCALIB = TESTS / 'mpi_shared_calibration.py'
MULTIRUN = TESTS / 'mpi_multirun_placement.py'
EXAMPLES = REPO / 'psana/psana/gpu/examples'

# (tag, ranks, script, extra argv, one_gpu, expect[, timeout])
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
# expect='hang-known'
#                 the job must hang at a KNOWN pre-existing bug, which the
#                 case is the reproducer for. Passing means the bug is still
#                 there; the case FAILS if the job completes, because that
#                 means the bug is fixed and the case should become 'clean'.
#                 See psana/psana/gpu/docs/multirun_hang.md.
#
# timeout defaults to DEFAULT_TIMEOUT and is given explicitly only for cases
# that legitimately take longer -- not as a way to accommodate a slow start,
# which is what an abort case's timing budget is there to catch.
DEFAULT_TIMEOUT = 180
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
    # Failure paths. The select-device case is the only test of the abort
    # reaching EB and smd0, which the device-group allreduce cannot touch.
    # Through the real DataSource -> Run -> GpuEventManager wiring. The other
    # placement cases call discover_peers directly, so they cover the
    # algorithm but not whether its result reaches a running job.
    ('realpath-2-gpus',      5, REALPATH, ['--expect-devices', '2'], False, 'clean'),
    # 6 ranks with PS_EB_NODES=1 is 1 smd0 + 1 eb + 4 bd; one visible GPU
    # makes all four peers, which is the sharing-relevant layout.
    ('realpath-1-gpu',       6, REALPATH,
     ['--expect-devices', '1', '--expect-peers', '4'], True, 'clean'),
    # Real Jungfrau calibration while the constants are shared over IPC, with
    # every panel still compared to a CPU reference. A follower calibrates
    # from the owner's imported mapping, so a shared view with a wrong offset
    # or a stale generation shows up as mismatched pixels.
    ('shared-calibration',   4, SHAREDCALIB,
     ['--user-dir', str(EXAMPLES), '--expect-peers', '2', '--require-sharing'],
     True, 'clean'),
    # Per-run rediscovery: run N+1 must not be sized against a card that
    # still holds run N's memory. This is the coverage review findings N1, N2
    # and finding 1 asked for -- and it cannot pass yet, because psana2 hangs
    # at the SECOND run of any multi-run MPI job. That is pre-existing on this
    # branch's parent (features/psana2-gpu) and reproduces with no GPU code
    # involved; measurements in psana/psana/gpu/docs/multirun_hang.md.
    #
    # Kept as 'hang-known' rather than deleted: it is the reproducer, and it
    # becomes the rediscovery test the moment multi-run works. The timeout is
    # short because a hang is what is expected -- there is no point paying 420s
    # for it.
    #
    # mfx101629726 rather than mfx100848724, which has only r0051, and not
    # mfx100852324, whose stream 7 describes jungfrau segments twice under the
    # `config` alg -- detector_field_handles refuses that, so setup aborts
    # before any event is read. See the driver's docstring.
    ('multirun-3-runs',      6, MULTIRUN,
     ['--exp', 'mfx101629726', '--runs', '15,16,17'], True, 'hang-known', 150),
    ('fail-select-device',   5, FAILURES, ['--inject', 'select-device'], False, 'abort'),
    ('fail-follower-import', 4, FAILURES, ['--inject', 'follower-import'], True, 'clean'),
)


def run_case(tag, ranks, script, extra, one_gpu, expect='clean',
             timeout=DEFAULT_TIMEOUT):
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
    elif expect == 'hang-known':
        ok, detail = judge_known_hang(text, timed_out)
    elif timed_out:
        ok, detail = False, {'error': f'timed out after {timeout}s'}
    else:
        ok, detail = parse(text)
        # The log alone is not enough in the other direction either: a crash
        # AFTER the rank records are written -- in teardown, say, where close()
        # frees shared allocations -- leaves a log full of passes. Require a
        # clean exit as well. Passing runs already report exit=0.
        if ok and status != 0:
            ok = False
            detail['error'] = (f'every rank passed but mpirun exited {status}; '
                               'something failed after the records were '
                               'written, most likely in teardown')
    detail['exit'] = status
    detail['elapsed'] = elapsed
    # Both conditions are required: mpirun exits 0 even when ranks report
    # failures, and the ranks can all report success before a late crash.
    print(f'CASE_END {tag} ok={ok} exit={status} elapsed={elapsed}s '
          f'{json.dumps(detail, sort_keys=True)}', flush=True)
    return ok, detail


def judge_known_hang(text, timed_out):
    """A case that reproduces a known pre-existing hang.

    It passes while the bug is present and fails once it is gone, so the
    suite tells you when the case should be promoted to 'clean' instead of
    quietly carrying a test that proves nothing.

    The hang must also look like the KNOWN one -- the first run processed,
    the second starting and not finishing. A job that hung before delivering
    anything is a different failure and is reported as such.
    """
    detail = {'expected': 'hang-known', 'timed_out': timed_out,
              'doc': 'psana/psana/gpu/docs/multirun_hang.md'}
    runs = sorted(set(re.findall(r'MULTIRUN_RANK\d+_RUN(\d+)', text)))
    detail['runs_reached'] = runs
    if not timed_out:
        detail['error'] = ('the job COMPLETED. The multi-run hang this case '
                           'reproduces appears to be fixed -- promote the '
                           'case to expect=clean and delete '
                           'docs/multirun_hang.md')
        return False, detail
    if len(runs) < 2:
        detail['error'] = (f'hung having reached runs {runs}; the known bug '
                           'processes the first run and stalls in the second, '
                           'so this is a different failure')
        return False, detail
    detail['note'] = (f'hung in run {runs[-1]} after processing {runs[0]}, '
                      'which is the known pre-existing failure')
    return True, detail


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

    for block in re.findall(r'SHAREDCALIB_SUMMARY (\{.*?\n\})', text, re.S):
        record = json.loads(block)
        detail['shared_calibration'] = {
            'workers': record['gpu_workers'],
            'devices': record['devices'],
            'shared_mib': record['shared_mib'],
            'saved_mib': record['device_memory_saved_mib'],
            'events': record['events_checked'],
            'all_pass': record['all_pass']}
        if not record['all_pass']:
            detail.setdefault('failures', []).extend(record['failures'][:5])

    for block in re.findall(r'REALPATH_SUMMARY (\{.*?\n\})', text, re.S):
        record = json.loads(block)
        detail['realpath'] = {'gpu_workers': record['gpu_workers'],
                              'devices': record['distinct_devices'],
                              'peers_by_device': record['peers_by_device'],
                              'claims': record['aggregate_claims'],
                              'all_pass': record['all_pass']}
        if not record['all_pass']:
            detail.setdefault('failures', []).extend(record['failures'][:5])

    for block in re.findall(r'MULTIRUN_SUMMARY (\{.*?\n\})', text, re.S):
        record = json.loads(block)
        detail['multirun'] = {'runs_sized': record['runs_sized'],
                              'usable_gib_by_rank': record['usable_gib_by_rank'],
                              'events_by_run': record['events_by_run'],
                              'calls_by_run': record['calls_by_run'],
                              'all_pass': record['all_pass']}
        if not record['all_pass']:
            detail.setdefault('failures', []).extend(record['failures'][:5])

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
    if 'realpath' in detail:
        ok = ok and detail['realpath']['all_pass']
    if 'shared_calibration' in detail:
        ok = ok and detail['shared_calibration']['all_pass']
    if 'multirun' in detail:
        ok = ok and detail['multirun']['all_pass']
    return ok, detail


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f'LOG_DIRECTORY {LOGS}', flush=True)
    results = {}
    for case in CASES:
        tag, ranks, script, extra, one_gpu, expect = case[:6]
        timeout = case[6] if len(case) > 6 else DEFAULT_TIMEOUT
        ok, _ = run_case(tag, ranks, script, extra, one_gpu, expect=expect,
                         timeout=timeout)
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

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
import shutil
import subprocess
import sys
import time


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
TESTS = REPO / 'psana/psana/tests/gpu/integration'
LOGS = HERE / f'cases-{time.strftime("%Y%m%d-%H%M%S")}'
# MPI session directories, which are large and transient, go somewhere
# without a quota. SCRATCH if the site sets it, else /tmp.
SESSIONS = pathlib.Path(
    os.environ.get('SCRATCH') or f'/tmp/{os.environ.get("USER", "psana")}'
) / f'ipc-sessions-{time.strftime("%Y%m%d-%H%M%S")}'

DISCOVERY = TESTS / 'mpi_placement_discovery.py'
FAILURES = TESTS / 'mpi_failure_paths.py'
MULTIRUN = TESTS / 'mpi_multirun_placement.py'
CALIB = TESTS / 'mpi_calib_pixel_exact.py'

# (tag, ranks, script, extra argv, one_gpu, expect[, timeout])
#
# one_gpu restricts the mask so every worker is a peer, which is what sharing
# needs; discovery deliberately spans whatever the launcher provided. It may
# also be a mask STRING, used verbatim -- that is how the reversed-mask case
# is expressed.
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
    # THE value check, in two layouts. A real psana job -- smd0, EBs and BD
    # workers as psana assigns them -- calibrates real Jungfrau data on the
    # GPU and compares every event against det.raw.calib() on the CPU.
    #
    # It asserts on PIXELS, not on internals. Device discovery, budget
    # sizing, constant sharing over IPC and batch routing have all already
    # happened by the time a pixel exists, so if any of them were wrong the
    # values would be wrong. That is why the white-box drivers this replaced
    # -- which probed _task_constants, placement records and peer counts --
    # are gone.
    #
    # one_gpu=True makes every BD worker a peer, so constants are shared from
    # a single device allocation; the 9-rank case leaves the mask wide, so the
    # workers split across cards and most groups are solo.
    #
    # 420s, not the 180s default. hybrid_det mirrors the bigdata streams
    # through BOTH the CPU and GPU paths -- that is what makes
    # det.raw.calib() available for the comparison -- so the I/O is done
    # twice. Measured standalone at ~175s per case on an idle A100, which the
    # default timeout clipped.
    ('calib-shared',         4, CALIB, ['--expect-events', '13'], True,
     'clean', 420),
    ('calib-2-gpus-eb2',     9, CALIB, ['--expect-events', '13'], False,
     'clean', 420),
    # Failure paths. The select-device case is the only test of the abort
    # reaching EB and smd0, which the device-group allreduce cannot touch.
    # A REVERSED mask. CUDA numbers visible devices in the order the mask
    # lists them, so with '1,0' the card that was physical 1 becomes ordinal
    # 0 and vice versa. Assignment keys on the PCI bus id the driver reports,
    # not on the ordinal, so the pixels must still come out right -- a rank
    # whose constants or batches followed the ordinal rather than the device
    # would calibrate from the wrong card's memory.
    #
    # Replaces masked_device.sbatch, which tested this as a standalone job.
    ('calib-reversed-mask',  6, CALIB, ['--expect-events', '13'], '1,0',
     'clean', 420),
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
    # An IPC failure now aborts rather than degrading to private copies. Still
    # expect='clean': the driver CATCHES the SharedConstantsError and asserts
    # the things an exit status cannot show -- that every peer raised and not
    # just the rank whose import failed, that the message names the opt-out,
    # and that nothing was left mapped or charged. A nonzero exit would only
    # say some rank died.
    ('fail-follower-import', 4, FAILURES, ['--inject', 'follower-import'], True, 'clean'),
)


def run_case(tag, ranks, script, extra, one_gpu, expect='clean',
             timeout=DEFAULT_TIMEOUT):
    """Launch one MPI job in its own session and return (ok, detail)."""
    env = os.environ.copy()
    # A private PRRTE session per case: without this, a case that aborts can
    # leave state that prevents the next `mpirun` from starting.
    #
    # Placed on scratch, NOT beside the logs. Open MPI's shared-memory BTL
    # wants a 128 MiB backing file per rank here, and $HOME is quota'd: when
    # it filled, every case died with "create_and_attach: unable to create
    # shared memory BTL coordinating structure" before reaching psana, which
    # reads as a hang rather than as a full disk. $SCRATCH or /tmp both have
    # room; the logs stay on home where they are small and wanted.
    session = SESSIONS / f'session-{tag}'
    session.mkdir(parents=True, exist_ok=True)
    env['TMPDIR'] = str(session)
    env['OMPI_MCA_orte_tmpdir_base'] = str(session)
    env['PMIX_MCA_ptl_tcp_if_include'] = env.get('PMIX_MCA_ptl_tcp_if_include', 'lo')
    # one_gpu is either a bool -- True narrows to the first card, so every
    # worker is a peer and sharing engages -- or a mask string used verbatim.
    # A string is how a REVERSED mask is expressed: CUDA numbers visible
    # devices in the order the mask lists them, so '1,0' makes ordinal 0 the
    # card that was physical 1. Narrowing to one card proves nothing about
    # that mapping, because every wrong implementation also lands on ordinal 0.
    if isinstance(one_gpu, str):
        env['CUDA_VISIBLE_DEVICES'] = one_gpu
    elif one_gpu:
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

    Timing out is not enough on its own -- the timeout is deliberately short,
    so an unrelated stall (a slow filesystem, a loaded node) would otherwise
    be reported as the known failure. The log must show the KNOWN SHAPE:

    * the first run delivered events,
    * the last run reached delivered none,
    * and no rank emitted its post-loop record, which would mean the run loop
      finished and the hang is somewhere else.
    """
    detail = {'expected': 'hang-known', 'timed_out': timed_out,
              'doc': 'psana/psana/gpu/docs/multirun_hang.md'}
    # (runnum, events) per rank per run. Sorted numerically: a lexicographic
    # sort reverses runs that cross a digit-width boundary, e.g. 9 and 10.
    seen = {}
    for block in re.findall(r'MULTIRUN_RANK\d+_RUN\d+ (\{.*?\n\})', text, re.S):
        try:
            record = json.loads(block)
        except ValueError:
            continue
        seen.setdefault(int(record['runnum']), []).append(
            int(record.get('events') or 0))
    runs = sorted(seen)
    detail['runs_reached'] = runs
    detail['events_by_run'] = {run: sum(seen[run]) for run in runs}

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
    # A rank that finished the run loop emits a record with no _RUN suffix.
    # Its presence means the loop completed and the hang is elsewhere.
    if re.search(r'MULTIRUN_RANK\d+ \{', text):
        detail['error'] = ('a rank finished the run loop, so the job did not '
                           'stall inside a run; this is a different failure '
                           'from the one documented')
        return False, detail
    first, last = runs[0], runs[-1]
    if not detail['events_by_run'][first]:
        detail['error'] = (f'run {first} delivered no events, so the job '
                           'stalled before the documented point rather than '
                           'between runs')
        return False, detail
    if detail['events_by_run'][last]:
        detail['error'] = (f'run {last} delivered '
                           f'{detail["events_by_run"][last]} events; the known '
                           'bug delivers none to the second run, so this is a '
                           'different failure')
        return False, detail
    detail['note'] = (f'processed run {first} '
                      f'({detail["events_by_run"][first]} events), then hung '
                      f'in run {last} with none delivered -- the known '
                      'pre-existing failure')
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



    for block in re.findall(r'CALIB_SUMMARY (\{.*?\n\})', text, re.S):
        record = json.loads(block)
        detail['calib'] = {'gpu_workers': record['gpu_workers'],
                           'events': record['events_total'],
                           'by_rank': record['events_by_rank'],
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


    # A driver that wraps device_comm in the test recorder (see
    # psana/psana/tests/gpu/collective_recorder.py) has each peer compare its
    # collective sequence with the others. A divergence is reported rather
    # than raised, so the launcher has to fail the case on it or the check is
    # advisory. Kept here because the drivers may enable it even though
    # production no longer reads an environment variable for it.
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
    if 'calib' in detail:
        ok = ok and detail['calib']['all_pass']
    if 'multirun' in detail:
        ok = ok and detail['multirun']['all_pass']
    return ok, detail


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f'LOG_DIRECTORY {LOGS}', flush=True)
    print(f'SESSION_DIRECTORY {SESSIONS}', flush=True)
    results = {}
    try:
        for case in CASES:
            tag, ranks, script, extra, one_gpu, expect = case[:6]
            timeout = case[6] if len(case) > 6 else DEFAULT_TIMEOUT
            ok, _ = run_case(tag, ranks, script, extra, one_gpu,
                             expect=expect, timeout=timeout)
            results[tag] = ok
    finally:
        # Each case's session holds a 128 MiB shared-memory file per rank, so
        # a full suite leaves gigabytes behind. Removed whatever the outcome:
        # an aborted run is exactly when they are largest.
        shutil.rmtree(SESSIONS, ignore_errors=True)

    print('\n=============================== VERDICT ===============================',
          flush=True)
    for tag, ok in results.items():
        print(f'  {"PASS" if ok else "FAIL"}  {tag}', flush=True)
    overall = all(results.values())
    print(f'  OVERALL: {"PASS" if overall else "FAIL"}', flush=True)
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())

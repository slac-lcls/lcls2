"""Exercise retry exhaustion, delayed success, evidence and time limits without I/O."""
import copy
import json
from types import SimpleNamespace
import pytest
import storage
from test_audit_port import fixture
from audit import audit_case

FILES = ['/qualified/a', '/qualified/b']


def harness(monkeypatch, sequence, scan_seconds=1):
    clock = [0.]
    calls = []
    def sleep(seconds): clock[0] += seconds
    monkeypatch.setattr(storage, 'time', SimpleNamespace(monotonic=lambda:clock[0], sleep=sleep))
    def scan(files, directory, media, evict=False):
        assert files == FILES and str(directory) == '/qualified' and media == 'lustre'
        calls.append(evict)
        clock[0] += scan_seconds
        value = sequence[len(calls)-1]
        if isinstance(value, Exception): raise value
        return [dict(path=p, pages=100, resident=value, resident_pages=value) for p in files]
    monkeypatch.setattr(storage, 'cache', scan)
    return calls


@pytest.mark.parametrize('sequence,attempts', [([0,0],1),([1,1,0,0],2),([1,0,0,1,0,0],3)])
def test_cold_success_requires_two_zero_scans(monkeypatch,tmp_path,sequence,attempts):
    calls = harness(monkeypatch,sequence)
    path = tmp_path/'cache.json'
    receipt = storage.condition_cache(FILES,'/qualified','lustre',path)
    assert len(receipt['attempts']) == attempts
    assert calls == [True,False]*attempts
    assert json.loads(path.read_text()) == receipt
    assert [s['files'][0]['resident_pages'] for a in receipt['attempts'] for s in a['scans']] == sequence
    assert storage.validate_cache_gate(receipt,FILES) == receipt['attempts'][-1]['scans']


def test_three_failures_preserve_all_counts(monkeypatch,tmp_path):
    calls = harness(monkeypatch,[3,2,2,1,1,1])
    path = tmp_path/'cache.json'
    with pytest.raises(ValueError,match='three attempts exhausted'):
        storage.condition_cache(FILES,'/qualified','lustre',path)
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'failed' and len(calls) == 6
    assert [s['files'][0]['resident_pages'] for a in receipt['attempts'] for s in a['scans']] == [3,2,2,1,1,1]


def test_slow_scan_cannot_pass_after_deadline(monkeypatch,tmp_path):
    calls = harness(monkeypatch,[0],scan_seconds=181)
    path = tmp_path/'cache.json'
    with pytest.raises(ValueError,match='conditioning deadline'):
        storage.condition_cache(FILES,'/qualified','lustre',path)
    receipt = json.loads(path.read_text())
    assert calls == [True] and receipt['status'] == 'failed'
    assert receipt['attempts'][0]['scans'][0]['files'][0]['resident_pages'] == 0
    assert receipt['elapsed_seconds'] == 181


def test_deadline_during_retry_pause_stops_before_eviction(monkeypatch,tmp_path):
    calls = harness(monkeypatch,[1,1],scan_seconds=89.5)
    path = tmp_path/'cache.json'
    with pytest.raises(ValueError,match='conditioning deadline'):
        storage.condition_cache(FILES,'/qualified','lustre',path)
    receipt = json.loads(path.read_text())
    assert calls == [True,False] and len(receipt['attempts']) == 1
    assert receipt['status'] == 'failed'


def test_scan_errors_fail_without_retry_and_keep_prior_evidence(monkeypatch,tmp_path):
    calls = harness(monkeypatch,[2,OSError('mincore failed')])
    path = tmp_path/'cache.json'
    with pytest.raises(OSError,match='mincore failed'):
        storage.condition_cache(FILES,'/qualified','lustre',path)
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'failed' and calls == [True,False]
    assert len(receipt['attempts'][0]['scans']) == 1
    assert receipt['attempts'][0]['scans'][0]['files'][0]['resident_pages'] == 2


@pytest.mark.parametrize('damage', ['missing','duplicate','negative','overfull','warm','one_scan','policy',
                                   'too_many','sequence','deadline','nan','timing','size_change',
                                   'early_success','alias','duration','verify_gap','retry_gap','exact_deadline'])
def test_independent_audit_rejects_bad_cache_evidence(damage):
    args = list(fixture()); obs = args[0]['observation']; receipt = obs['cache_gate']
    scans = receipt['attempts'][0]['scans']; rows = scans[0]['files']
    if damage == 'missing': rows.pop()
    if damage == 'duplicate': rows[1]['path'] = rows[0]['path']
    if damage == 'negative': rows[0]['resident_pages'] = rows[0]['resident'] = -1
    if damage == 'overfull': rows[0]['resident_pages'] = rows[0]['resident'] = 101
    if damage == 'warm': rows[0]['resident_pages'] = rows[0]['resident'] = 1
    if damage == 'one_scan': scans.pop()
    if damage == 'policy': receipt['max_attempts'] = 4
    if damage == 'too_many': receipt['attempts'] *= 4
    if damage == 'sequence': receipt['attempts'][0]['number'] = 2
    if damage == 'deadline': receipt['elapsed_seconds'] = 181.
    if damage == 'nan': receipt['elapsed_seconds'] = float('nan')
    if damage == 'timing': scans[1]['elapsed_seconds'] = .5
    if damage == 'size_change': scans[1]['files'][0]['pages'] += 1
    if damage == 'early_success': receipt['attempts'].append(copy.deepcopy(receipt['attempts'][0]))
    if damage == 'alias': obs['initial_cache'][0]['resident_pages'] = 1
    if damage == 'duration': obs['conditioning_seconds'] += 1
    if damage == 'verify_gap': scans[1]['elapsed_seconds'] = scans[0]['elapsed_seconds']
    if damage == 'exact_deadline': receipt['elapsed_seconds'] = 180.
    if damage == 'retry_gap':
        first = copy.deepcopy(receipt['attempts'][0])
        for scan in first['scans']:
            for row in scan['files']: row['resident'] = row['resident_pages'] = 1
        receipt['attempts'][0]['number'] = 2
        scans[0]['elapsed_seconds'] = 2.
        scans[1]['elapsed_seconds'] = 2.2
        receipt['attempts'].insert(0,first)
        receipt['elapsed_seconds'] = obs['conditioning_seconds'] = 2.2
    with pytest.raises(ValueError): audit_case(*args)


def test_all_case_limits_include_conditioning_budget():
    import plan
    assert plan.ATTEMPT == 'comparison02'
    for case in plan.cases():
        base = 300 if case['kind']=='acceptance' else 240 if case['stage']=='S' else {1:600,4:360}[case['bds']]
        assert case['timeout'] == base + storage.CACHE_SECONDS
    assert plan.BUDGET['cache_conditioning'] == 35*180
    assert sum(plan.BUDGET.values()) == plan.WALLTIME == 21300


def test_receipt_write_delay_cannot_pass_deadline(monkeypatch,tmp_path):
    harness(monkeypatch,[0,0])
    original = storage.write
    def delayed_write(path, receipt):
        original(path, receipt)
        if receipt['status'] == 'pass': storage.time.sleep(180)
    monkeypatch.setattr(storage, 'write', delayed_write)
    path = tmp_path/'cache.json'
    with pytest.raises(ValueError,match='conditioning deadline'):
        storage.condition_cache(FILES,'/qualified','lustre',path)
    assert json.loads(path.read_text())['status'] == 'failed'


def test_independent_audit_accepts_third_attempt_success():
    args = list(fixture()); obs = args[0]['observation']; receipt = obs['cache_gate']
    template = receipt['attempts'][0]
    attempts = []
    for number in range(1,4):
        attempt = copy.deepcopy(template); attempt['number'] = number
        for index, scan in enumerate(attempt['scans']):
            scan['elapsed_seconds'] = 3*(number-1)+index+1.
            if number < 3:
                for row in scan['files']: row['resident'] = row['resident_pages'] = 1
        attempts.append(attempt)
    receipt['attempts'] = attempts
    receipt['elapsed_seconds'] = obs['conditioning_seconds'] = 8.
    assert audit_case(*args)['status'] == 'pass'

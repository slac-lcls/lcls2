"""Prevent partial, cross-node or mismatched timing comparisons."""
import copy
import pytest
from comparison import summarize
from plan import cases


def records():
    times = {'S':10., 'E':20., 'EC':30., 'ECC':40.}
    return [dict(case=c, seconds=times[c['stage']] * (2 if c['cohort']=='anchor' else 1),
                 audit='pass', job='job1', node='node1', cache_attempts=1, conditioning_seconds=2.) for c in cases()]


def test_only_full_comparison_cells_enter_stage_statistics():
    result = summarize(records())
    assert len(result['cache_conditioning']) == 35
    assert all(r['attempts'] == 1 and r['seconds'] == 2. for r in result['cache_conditioning'])
    assert [r['name'] for r in result['cache_conditioning']] == [c['name'] for c in cases()]
    assert len(result['comparison']) == 8
    assert len(result['paired']) == 18
    assert len(result['staging_drift']) == 6
    assert all(r['after_over_before'] == 2 for r in result['staging_drift'])
    assert all(r['seconds_over_staging'] == 4 for r in result['paired'] if r['stage']=='ECC')
    assert all(r['seconds']['median'] == 10 for r in result['comparison'] if r['stage']=='S')
    assert result['staging_bridge_seconds']['samples'] == [10.,10.,10.]


@pytest.mark.parametrize('damage', ['missing','duplicate','order','events','node','job','audit','nan','zero','cache_attempts','cache_seconds'])
def test_invalid_comparison_rejected(damage):
    rows = copy.deepcopy(records())
    if damage=='missing': rows.pop()
    if damage=='duplicate': rows[-1] = rows[-2]
    if damage=='order': rows[3],rows[4] = rows[4],rows[3]
    if damage=='events': rows[3]['case']['events'] = 1003
    if damage=='node': rows[3]['node'] = 'node2'
    if damage=='job': rows[3]['job'] = 'job2'
    if damage=='audit': rows[3]['audit'] = 'fail'
    if damage=='nan': rows[3]['seconds'] = float('nan')
    if damage=='zero': rows[3]['seconds'] = 0
    if damage=='cache_attempts': rows[3]['cache_attempts'] = 4
    if damage=='cache_seconds': rows[3]['conditioning_seconds'] = 180
    with pytest.raises(ValueError): summarize(rows)


@pytest.mark.parametrize('damage', [None, 'hash', 'audit', 'placement', 'memory', 'trace'])
def test_build_reaudits_saved_receipts(tmp_path, monkeypatch, damage):
    import json
    import comparison
    from support import write, sha
    from test_audit_port import fixture
    report, case, info, ranges, mapping, gpu = fixture()
    case['name'] = 'case1'
    report['job'] = 'job1'
    report['rate_is_clean'] = True
    report['rows'][2]['trace_window'] = dict(tracer_pid=0)
    if damage=='placement': report['rows'][2]['affinity'] = [99]
    if damage=='trace': report['rows'][2]['trace_window']['tracer_pid'] = 42
    output = tmp_path/'results'; output.mkdir()
    write(tmp_path/'inputs.json', info); write(tmp_path/'ranges.json', ranges)
    write(output/'hardware.json', dict(uuids=['GPU-'+str(i) for i in range(4)], pci_by_uuid=gpu['pci_by_uuid']))
    write(output/'affinity.json', dict(mapping=mapping))
    verdict = dict(status='pass', events=50, logical_payload_bytes=50*comparison.EVENT_BYTES)
    write(output/'case1.json', report)
    write(output/'status.json', dict(job='job1', source_commit='commit1', completed=[dict(case=case, audit=verdict, result_sha256=sha(output/'case1.json'))]))
    write(output/'case1.audit.json', verdict if damage!='audit' else dict(status='fail'))
    samples = [dict(time=t,total=40<<30,free=39<<30,used=1<<30) for t in (1.,2.)]
    memory = dict(uuid='GPU-0',samples=samples,errors=[],max_used=1<<30,min_free=39<<30)
    if damage=='memory': memory['uuid']='GPU-1'
    write(output/'case1.memory.json', [memory])
    if damage=='hash':
        report['seconds'] = 2.; write(output/'case1.json', report)
    monkeypatch.setattr(comparison, 'cases', lambda:[case])
    monkeypatch.setattr(comparison, 'summarize', lambda records:dict(status='pass',records=records))
    if damage:
        with pytest.raises(ValueError): comparison.build(tmp_path)
    else:
        result = comparison.build(tmp_path)
        assert result['records'][0]['seconds'] == report['seconds']
        assert result['records'][0]['cache_attempts'] == 1
        assert result['records'][0]['conditioning_seconds'] == 2.
        assert json.loads((output/'comparison.json').read_text())['source_commit']=='commit1'


def test_summary_exposes_retries_without_adjusting_processing_times():
    rows = records()
    rows[3]['cache_attempts'] = 3
    rows[3]['conditioning_seconds'] = 120.
    result = summarize(rows)
    assert result['cache_conditioning'][3]['attempts'] == 3
    assert result['cache_conditioning'][3]['seconds'] == 120.
    assert result['comparison'][0]['seconds']['median'] == 10.

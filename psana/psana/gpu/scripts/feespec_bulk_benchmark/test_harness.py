import json
import pytest
from common import tier
from run import matrix
from common import prefix_residency


def test_two_rounds_reverse_variant_and_cache_order():
    rows=list(matrix())
    assert len(rows)==len(set(rows))==12
    assert rows[:3]==[('A','cold',1),('E-off','cold',1),('E-on','cold',1)]
    assert rows[6:9]==[('E-on','warm',2),('E-off','warm',2),('A','warm',2)]
    assert all(sum(v==variant and c==cache for v,c,_ in rows)==2
               for variant in ('A','E-off','E-on') for cache in ('cold','warm'))


@pytest.mark.parametrize('ssd,object_bytes,remote,accepted',[
    (4096,0,0,True),(4096,4096,0,False),(2048,0,0,False),(4096,0,4096,False)])
def test_tier_requires_full_ssd_and_no_object_or_remote_backing(monkeypatch,ssd,object_bytes,remote,accepted):
    value=[dict(path='/private/ffb/input',file_size=4096,ssd_write_cache_bytes=ssd,
                ssd_read_cache_bytes=0,object_storage_bytes=object_bytes,remote_storage_bytes=remote)]
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:json.dumps(value))
    if accepted:
        assert tier(['/private/ffb/input'])==value
    else:
        with pytest.raises(RuntimeError):tier(['/private/ffb/input'])


def test_short_run_measures_only_selected_prefix(tmp_path):
    import mmap
    path=tmp_path/'input'
    path.write_bytes(b'x'*(mmap.PAGESIZE*4))
    row=prefix_residency(path,mmap.PAGESIZE)
    assert row['bytes']==mmap.PAGESIZE and row['pages']==1
    assert row['resident_pages']==1
    with pytest.raises(ValueError):prefix_residency(path,mmap.PAGESIZE*5)
    with pytest.raises(ValueError):prefix_residency(path,0)


def test_short_baseline_balances_order():
    from quick import matrix as quick_matrix
    assert quick_matrix()==[('E-off',1),('E-on',1),('E-on',2),('E-off',2)]


def test_warm_preparation_reads_only_selected_prefix(tmp_path, monkeypatch):
    from pathlib import Path
    from common import cache_inputs
    path = tmp_path/'data.xtc2'
    path.write_bytes(bytes(32768))
    original = Path.open
    read_bytes = []
    class Source:
        def __enter__(self):
            self.source = original(path, 'rb', buffering=0)
            return self
        def __exit__(self, *args):
            self.source.close()
        def read(self, n):
            result = self.source.read(n)
            read_bytes.append(len(result))
            return result
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: Source())
    monkeypatch.setattr('common.prefix_residency', lambda *args: dict(
        pages=2, resident_pages=2, resident_fraction=1.))
    result = cache_inputs(tmp_path, 'warm', ranges={path.name: 8192})
    assert sum(read_bytes) == 8192 and result['resident_fraction'] == 1.


def test_warm_prefix_gate_rejects_missing_pages(tmp_path, monkeypatch):
    from common import cache_inputs
    path = tmp_path/'data.xtc2'
    path.write_bytes(bytes(4096))
    monkeypatch.setattr('common.prefix_residency', lambda *args: dict(
        pages=1, resident_pages=0, resident_fraction=0.))
    with pytest.raises(RuntimeError, match='Warm prefix'):
        cache_inputs(tmp_path, 'warm', prepare=False, ranges={path.name: 4096})


def test_acceptance_keeps_trace_rates_out_of_verdict(tmp_path):
    from acceptance import matrix, summarize
    cases = matrix()
    assert len(cases) == len(set(cases)) == 20
    rows = []
    for build, variant, cache, repetition, mode in cases:
        seconds = (12 if variant == 'E-on' else 10) if mode == 'control' else .01
        rows.append(dict(build=build, variant=variant, cache=cache,
                         repetition=repetition, trace_mode=mode, events=1000,
                         loop_s=seconds, events_per_s=1000/seconds,
                         fallback_audit=dict(operations={'POSIX': {'calls': 38000}},
                             file_concurrency=dict(single_file_s=1, posix_active_s=10,
                                                   single_file_percent=10))))
    summarize(tmp_path, rows, complete=True)
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['complete'] and not report['performance_target_met']
    assert all(c['current_on_off_time_ratio'] == 1.2 for c in report['comparisons'].values())


def test_depth_comparison_has_two_rounds_and_excludes_diagnostics(tmp_path):
    from depth_summary import matrix, summarize
    cases = matrix()
    assert len(cases) == len(set(cases)) == 28
    assert sum(c[-1] == 'control' for c in cases) == 16
    assert sum(c[-1] == 'trace' for c in cases) == 8
    rows = []
    for build, variant, cache, repetition, mode in cases:
        seconds = (10 if build == 'depth1' else 8) if mode == 'control' else 100
        rows.append(dict(build=build, variant=variant, cache=cache,
            repetition=repetition, trace_mode=mode, loop_s=seconds, events_per_s=1000/seconds,
            ranks=[{}, {}, dict(counts=dict(read_wait_s=2), pipeline_stats=dict(
                subbatches=[100]*10, launches={}, peak_charged_bytes=4096, allocation_reserve_calls=1))],
            fallback_audit=dict(file_concurrency=dict(single_file_percent=10, posix_active_s=1),
                                no_posix_read_s=99, no_posix_read_percent=99)))
    summarize(tmp_path, rows, complete=True)
    result = json.loads((tmp_path/'summary.json').read_text())
    assert result['complete']
    assert list(result['depth2_depth1_time_ratios'].values()) == [.8]*4


def test_four_mib_comparison_has_two_rounds_and_excludes_diagnostics(tmp_path):
    from acceptance import matrix, summarize
    cases = matrix('size4')
    assert len(cases) == len(set(cases)) == 14
    assert sum(c[-1] == 'control' for c in cases) == 8
    assert sum(c[-1] == 'trace' for c in cases) == 4
    rows = []
    for build, variant, cache, repetition, mode in cases:
        seconds = (10 if variant == 'E-off' else 12) if mode == 'control' else 100
        rows.append(dict(build=build, variant=variant, cache=cache,
            repetition=repetition, trace_mode=mode, loop_s=seconds, events_per_s=1000/seconds,
            ranks=[{}, {}, dict(counts=dict(read_wait_s=2), pipeline_stats=dict(
                subbatches=[100]*10, launches={}, peak_charged_bytes=4096, allocation_reserve_calls=1))],
            fallback_audit=dict(file_concurrency=dict(single_file_percent=10, posix_active_s=1),
                                no_posix_read_s=99, no_posix_read_percent=99)))
    summarize(tmp_path, rows, complete=True, study='size4')
    result = json.loads((tmp_path/'summary.json').read_text())
    assert result['complete']
    assert result['bulk_on_off_time_ratios'] == {'cold': 1.2, 'warm': 1.2}


def test_profile_call_graph_and_control_separation(tmp_path):
    import cProfile
    from acceptance import matrix, summarize
    from profile_summary import read_profile

    def leaf():
        return sum(range(100))

    def parent():
        leaf()
        leaf()

    profiler = cProfile.Profile()
    profiler.runcall(parent)
    path = tmp_path/'loop.pstats'
    profiler.dump_stats(path)
    profile = read_profile(path)
    item, = [r for r in profile['functions'] if r['function'] == 'leaf']
    assert item['calls'] == 2
    assert item['callers'][0]['function'] == 'parent'
    assert item['callers'][0]['calls'] == 2
    cases = matrix('profile')
    assert len(cases) == len(set(cases)) == 16
    assert cases[0][1:] == ('E-off', 'cold', 1, 'control')
    assert cases[8][1:] == ('E-on', 'warm', 2, 'profile')
    rows = [dict(build=b, variant=v, cache=c, repetition=r, trace_mode=m,
                 loop_s=10 if m == 'control' else 100,
                 events_per_s=100 if m == 'control' else 10,
                 python_profile=profile)
            for b, v, c, r, m in cases]
    summarize(tmp_path, rows, complete=True, study='profile')
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['complete'] and report['accepted_samples'] == 16
    assert all(v == dict(control=10, profile=100) for v in report['medians'].values())


@pytest.mark.parametrize('study', ['slots', 'files', 'groups'])
def test_slots_comparison_is_bulk_on_only_and_checks_pipeline_equivalence(tmp_path, study):
    from acceptance import matrix, summarize
    cases = matrix(study)
    assert len(cases) == len(set(cases)) == 14
    assert {c[1] for c in cases} == {'E-on'}
    assert [sum(c[-1] == mode for c in cases) for mode in ('control', 'profile', 'pipeline')] == [8, 4, 2]
    rows = []
    for build, variant, cache, repetition, mode in cases:
        seconds = (10 if build == 'baseline' else 9) if mode == 'control' else 100
        rows.append(dict(build=build, variant=variant, cache=cache, repetition=repetition,
            trace_mode=mode, loop_s=seconds, events_per_s=1000/seconds,
            python_profile=dict(functions=[dict(file='/psana/gpu/gpu_input_group.py',
                function='plan_slots', calls=31, self_s=.01, cumulative_s=.6),
                dict(file='/psana/gpu/gpu_kvikio_read.py', function='_prune_files',
                     calls=5019, self_s=.02, cumulative_s=.4),
                dict(file='/psana/gpu/gpu_kvikio_read.py', function='issue_group',
                     calls=5019, self_s=.04, cumulative_s=.9)]),
            ranks=[{}, {}, dict(pipeline_stats=dict(subbatches=[100]*10, launches={},
                peak_charged_bytes=4096, allocation_reserve_calls=1))]))
    summarize(tmp_path, rows, complete=True, study=study)
    result = json.loads((tmp_path/'summary.json').read_text())
    assert result['time_ratios'] == dict(cold=.9, warm=.9)
    assert result['pipeline_equivalent'] and result['complete']
    assert {p['function'] for p in result['profiles']} == {
        dict(slots='plan_slots', files='_prune_files', groups='issue_group')[study]}
    rows[-1]['ranks'][2]['pipeline_stats']['peak_charged_bytes'] += 512
    with pytest.raises(AssertionError, match='pipeline counts or charged-memory'):
        summarize(tmp_path, rows, complete=True, study=study)


@pytest.mark.parametrize('study', ['slots-warm', 'files-warm', 'groups-warm', 'groups-cold'])
def test_slots_warm_repeat_balances_four_pairs_and_needs_no_diagnostic_rows(tmp_path, study):
    from acceptance import matrix, summarize
    cases = matrix(study)
    assert len(cases) == len(set(cases)) == 8
    assert {r[1] for r in cases} == {'E-on'}
    cache = 'cold' if study.endswith('-cold') else 'warm'
    assert {r[2] for r in cases} == {cache}
    assert [r[0] for r in cases] == ['baseline', 'candidate', 'candidate', 'baseline'] * 2
    rows = [dict(build=b, variant=v, cache=c, repetition=r, trace_mode=m,
                 loop_s=10 if b == 'baseline' else 11, events_per_s=1000/(10 if b == 'baseline' else 11))
            for b, v, c, r, m in cases]
    summarize(tmp_path, rows, complete=True, study=study)
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['complete'] and report['time_ratios'] == {cache: 1.1}
    assert report['pipeline_equivalent'] is None


def test_current_comparison_uses_requested_event_count_and_balances_order(tmp_path):
    from acceptance import matrix, summarize
    cases = matrix('current')
    assert len(cases) == len(set(cases)) == 8
    assert cases[0][1:] == ('E-off', 'cold', 1, 'control')
    assert cases[4][1:] == ('E-on', 'warm', 2, 'control')
    rows = [dict(build=b, variant=v, cache=c, repetition=r, trace_mode=m,
                 events=10000, loop_s=80 if v == 'E-on' else 100,
                 events_per_s=125 if v == 'E-on' else 100,
                 ranks=[{}, {}, dict(counts=dict(requests=50182 if v == 'E-on' else 60000))])
            for b, v, c, r, m in cases]
    summarize(tmp_path, rows, complete=True, study='current')
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['complete'] and report['bulk_on_off_time_ratios'] == dict(cold=.8, warm=.8)
    assert report['groups']['E-on/warm']['events_per_s'] == 125
    with pytest.raises(AssertionError):
        summarize(tmp_path, rows[:-1], complete=True, study='current')
    rows[-1]['events'] = 1000
    with pytest.raises(AssertionError):
        summarize(tmp_path, rows, complete=True, study='current')


def test_acceptance_validates_long_run_count_and_expected_reads():
    from acceptance import validate
    reference = dict(events=10000, timestamp_sha256='t', sums_sha256='s', payload_bytes=336000000000)
    state = dict(measured_prefixes=True, files=[dict(resident_fraction=1.)]*6)
    row = dict(reference, diagnostic=False, include_jf=True, cache='warm', variant='E-on',
               cache_before=state, cache_after=state,
               ranks=[dict(affinity=[0]), dict(affinity=[0]),
                      dict(affinity=[0], counts=dict(requests=50182, bytes=reference['payload_bytes']))])
    provenance = dict(reference=reference, affinity=[0],
                      builds=dict(current=dict(expected_requests={'E-on':50182})))
    validate(row, 'current', provenance)
    row['events'] = 1000
    with pytest.raises(AssertionError):
        validate(row, 'current', provenance)
    row['events'] = 10000
    row['ranks'][2]['counts']['requests'] = 5019
    with pytest.raises(AssertionError):
        validate(row, 'current', provenance)


def test_large_warm_prefixes_interleave_and_still_enforce_every_file_gate(tmp_path, monkeypatch):
    from common import cache_inputs
    paths = [tmp_path/f's{i}.xtc2' for i in range(2)]
    length = 32 * 1024**3
    for path in paths:
        with path.open('wb') as stream:
            stream.truncate(length + 4096)
    calls = []
    monkeypatch.setattr('common.subprocess.run', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr('common.prefix_residency', lambda *args: dict(
        pages=1, resident_pages=1, resident_fraction=1.))
    ranges = {p.name:length for p in paths}
    cache_inputs(tmp_path, 'warm', ranges=ranges)
    assert len(calls) == 1 and calls[0][:2] == ['numactl', '--interleave=all']
    assert calls[0][-2] == '--prefixes'
    assert json.loads(calls[0][-1]) == [[str(p), length] for p in paths]
    monkeypatch.setattr('common.prefix_residency', lambda p, n: dict(
        pages=100, resident_pages=98 if p == paths[0] else 100,
        resident_fraction=.98 if p == paths[0] else 1.))
    with pytest.raises(RuntimeError, match='Warm prefix'):
        cache_inputs(tmp_path, 'warm', prepare=False, ranges=ranges)
    assert len(calls) == 1


def test_interleaved_helper_reads_exact_prefix_and_rejects_short_input(tmp_path, monkeypatch):
    import builtins
    import io
    from warm_cache import warm_prefix
    class Source(io.BytesIO):
        def close(self):
            pass
    source = Source(bytes(32768))
    monkeypatch.setattr(builtins, 'open', lambda *a, **kw: source)
    warm_prefix(tmp_path/'input', 8192)
    assert source.tell() == 8192
    source.seek(0)
    with pytest.raises(RuntimeError, match='short warm-prefix'):
        warm_prefix(tmp_path/'input', 65536)

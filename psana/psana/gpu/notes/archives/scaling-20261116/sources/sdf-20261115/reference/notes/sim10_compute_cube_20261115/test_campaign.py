"""CPU-only campaign tests; no staging, submission, GPU use or resource cleanup."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load('compute_campaign_plan_test', 'plan.py')
with patch.dict(sys.modules, {'plan': plan}):
    prepare = load('compute_campaign_prepare_test', 'prepare.py')
    with patch.dict(sys.modules, {'prepare': prepare}):
        launch = load('compute_campaign_launch_test', 'launch.py')


def budget():
    return dict(qualification=600, staging=3600, acceptance=7200, final_audit=900, safety=600,
                cases={kind: dict(setup=120, processing=1200, export=180, reference=300, audit=120)
                       for kind in ('pilot', 'full', 'anchor')},
                basis=dict(kind='conservative_unmeasured', justification='Explicit compute and CPU reference ceilings, not a rate forecast'))


def fake_result(case, ownership):
    stamps = list(range(case['events']))
    cpu = dict(ranks=[[rank] for rank in range(case['bds'] + 2)])
    gpu = plan.placement(['GPU-%d' % index for index in range(4)], case['bds'])
    rows, offset = [], 0
    for rank in range(case['bds'] + 2):
        count = ownership[rank - 2] if rank >= 2 else 0
        delivered = stamps[offset:offset + count]
        offset += count
        rows.append(dict(rank=rank, host='sdfampere036', affinity=cpu['ranks'][rank], timestamps=delivered,
            device=dict(mask=gpu['rank_uuids'][rank]),
            callbacks=[dict(size=len(delivered[start:start + case['batch']]),
                            timestamps=delivered[start:start + case['batch']]) for start in range(0, count, case['batch'])],
            io=dict(requests=5 * count, completed_requests=5 * count, requested_bytes=plan.EVENT_BYTES * count,
                    completed_bytes=plan.EVENT_BYTES * count, pressure_drains=0,
                    read_size_histogram={'524724': count, '629664': count, '734604': count * 3} if count else {}),
            cpu_reads=0, errors=0, trace_window=dict(tracer_pid=99 if case['kind'] == 'pilot' else 0),
            task_copy=dict(ordinary_host_fallbacks=0, bytes=0 if case['stage'] == 'S' else count * 128),
            acceptance=[dict(timestamp=stamp) for stamp in delivered]))
    result = dict(status='pass', stage=case['stage'], kind=case['kind'], events=case['events'],
                  hz=100.0, seconds=case['events'] / 100, rate_is_clean=case['kind'] in ('full', 'anchor'), rows=rows,
                  observation=dict(storage_evidence=dict(idle_adjusted_rx_payload_ratio=1.0)),
                  cube=dict(status='complete', global_events=case['events'], nbins=100, active_bins=99),
                  numerical_checks=[dict(bin=index, count=len(stamps[index::99]) if index < 99 else 0,
                                         pixels=32 * 512 * 1024, max_bound_ratio=0.0) for index in range(100)])
    return result, stamps, cpu, gpu


class CampaignTests(unittest.TestCase):
    def test_exact_matrix_and_order(self):
        cases = plan.matrix()
        self.assertEqual(Counter(case['kind'] for case in cases), {'pilot': 8, 'full': 24, 'anchor': 6})
        self.assertEqual(len({case['name'] for case in cases}), 38)
        self.assertEqual(cases, plan.matrix())
        self.assertTrue(all(case['media'] == 'weka' for case in cases))
        for offset in range(8, 38, 5):
            block = cases[offset:offset + 5]
            self.assertEqual({case['stage'] for case in block[:4]}, set(plan.STAGES))
            self.assertEqual((block[-1]['kind'], block[-1]['stage']), ('anchor', 'S'))
            self.assertEqual(len({case['bds'] for case in block}), 1)
        self.assertEqual([cases[offset]['bds'] for offset in range(8, 38, 5)], [1, 4, 4, 1, 1, 4])
        self.assertEqual([cases[offset]['stage'] for offset in (8, 18, 28)], ['S', 'E', 'EC'])

    def test_acceptance_tail_cases(self):
        self.assertEqual([(case['bds'], case['batch'], case['events']) for case in plan.acceptance_cases()],
                         [(1, 50, 201), (4, 200, 1003)])

    def test_placement_and_cpu_demand(self):
        uuids = ['GPU-%d' % index for index in range(4)]
        self.assertEqual(plan.placement(uuids, 1)['rank_uuids'], ['', '', 'GPU-0'])
        self.assertEqual(plan.placement(uuids, 4)['rank_uuids'], ['', ''] + uuids)
        with self.assertRaises(ValueError):
            plan.placement(uuids[:3], 4)
        with plan.historical() as helpers:
            topology = [dict(cpu=index, socket=index % 2, core=index // 2) for index in range(36)]
            self.assertEqual(helpers.cpu_plan.plan(topology, 1, 4)['required_physical_cores'], 13)
            self.assertEqual(helpers.cpu_plan.plan(topology, 4, 4)['required_physical_cores'], 34)

    def test_task_local_imports_restore_aliases_and_search_path(self):
        sentinel = object()
        search = sys.path[:]
        with patch.dict(sys.modules, {'plan': sentinel}):
            with plan.historical() as helpers:
                self.assertEqual(helpers.storage.ROOTS['weka'], plan.WEKA)
                self.assertIsNot(sys.modules['plan'], sentinel)
            self.assertIs(sys.modules['plan'], sentinel)
        self.assertEqual(sys.path, search)

    def test_changed_historical_helper_rejected_before_import(self):
        with patch.object(plan, 'sha', return_value='0' * 64), self.assertRaisesRegex(ValueError, 'Historical helper changed'):
            with plan.historical():
                self.fail('Changed helper imported')

    def test_runtime_receipt_exact_hash(self):
        with patch.object(prepare, 'sha', return_value='bad'), self.assertRaisesRegex(ValueError, 'receipt changed'):
            prepare.native_dependencies()

    def test_all_case_commands_match_parent_interface(self):
        for case in plan.acceptance_cases() + plan.matrix():
            command = launch.case_command(case, '/inputs.json', '/output.json', '/lscratch/monarin/cube')
            self.assertEqual(command[command.index('-n') + 1], str(case['bds'] + 2))
            self.assertEqual(command[command.index('--batch') + 1], str(case['batch']))
            self.assertEqual(str(plan.BASE / 'trace_runner.py') in command, case['kind'] == 'pilot')
            self.assertNotIn('sbatch', command)
            self.assertNotIn('--workers', command)
            self.assertEqual(command[-2:], ['--local-output', '/lscratch/monarin/cube'])

    def test_reference_export_and_partition_budgets(self):
        estimate = budget()
        total = plan.validate_budget(estimate, 100000, 864000)
        with self.assertRaisesRegex(ValueError, 'Budget exceeds'):
            plan.validate_budget(estimate, total - 1, 864000)
        with self.assertRaisesRegex(ValueError, 'Budget exceeds'):
            plan.validate_budget(estimate, 100000, 86400)
        estimate['cases']['pilot'].pop('reference')
        with self.assertRaisesRegex(ValueError, 'reference/export'):
            plan.validate_budget(estimate, 100000, 864000)

    def test_budget_rejects_staging_extrapolation_and_missing_ceiling(self):
        estimate = budget()
        estimate['basis']['kind'] = 'staging_extrapolation'
        with self.assertRaises(ValueError):
            plan.validate_budget(estimate, 100000, 864000)
        estimate = budget()
        estimate['cases']['full']['export'] = 0
        with self.assertRaises(ValueError):
            plan.validate_budget(estimate, 100000, 864000)

    def test_remaining_gate_reserves_all_work(self):
        estimate = budget()
        required = launch.remaining_gate(100000, estimate, plan.matrix(), now=0)
        self.assertGreater(required, sum(plan.case_seconds(estimate, case) for case in plan.matrix()))
        with self.assertRaisesRegex(ValueError, 'Incomplete campaign'):
            launch.remaining_gate(required - 1, estimate, plan.matrix(), now=0)
        self.assertEqual(launch.remaining_gate(required, estimate, plan.matrix(), now=0), required)

    def test_retain_every_cube_not_just_last_repeat(self):
        plan.validate_retention(plan.DEFAULT_RETENTION)
        expected = 10 * plan.EXPORT_ALLOWANCE + (14 << 30)
        self.assertEqual(plan.retained_bytes(plan.matrix(), plan.DEFAULT_RETENTION), expected)
        capacity = deepcopy(plan.DEFAULT_RETENTION)
        capacity['acceptance_bytes'] = 1
        with self.assertRaisesRegex(ValueError, 'acceptance cubes'):
            plan.validate_retention(capacity)

    def test_local_output_cannot_fall_back_to_shared(self):
        with self.assertRaisesRegex(ValueError, 'Local cube directory only'):
            launch.local_capacity(plan.SCRATCH, 1)

    def test_partition_time_parser(self):
        self.assertEqual(launch.slurm_seconds('10-00:00:00'), 864000)
        self.assertEqual(launch.slurm_seconds('24:00:00'), 86400)
        with self.assertRaises(ValueError):
            launch.slurm_seconds('UNLIMITED')

    def test_parent_availability_preserves_unavailable_state(self):
        record = prepare.partition_record(dict(partition_max_time='10-00:00:00',
            commands=['scontrol show partition ampere'], ready_immediately=False))
        self.assertEqual(record['max_time_seconds'], 864000)
        self.assertFalse(record['ready_immediately'])

    def test_modern_exclusive_node_field(self):
        self.assertEqual(launch.exclusive_gate({'Exclusive': 'NODE', 'OverSubscribe': 'NO'}, {}, ['123'], '123'),
                         'Exclusive=NODE')
        for value in ('USER', 'TOPO', 'NO'):
            with self.assertRaises(ValueError):
                launch.exclusive_gate({'Exclusive': value}, {}, ['123'], '123')

    def test_slurm_25_11_no_requires_whole_node_corrobation(self):
        fields = dict(OverSubscribe='NO', NumCPUs='112')
        node = dict(CPUEfctv='112', CPUAlloc='112')
        self.assertIn('25.11', launch.exclusive_gate(fields, node, ['123'], '123'))
        with self.assertRaises(ValueError):
            launch.exclusive_gate(fields, node, ['123', '456'], '123')
        with self.assertRaises(ValueError):
            launch.exclusive_gate(dict(fields, NumCPUs='36'), node, ['123'], '123')
        with self.assertRaises(ValueError):
            launch.exclusive_gate({'OverSubscribe': 'NO'}, {}, ['123'], '123')

    def test_status_alone_is_not_execution_evidence(self):
        case = plan.matrix()[0]
        with self.assertRaises((ValueError, KeyError)):
            launch.validate_result(dict(status='pass', events=4000, stage='S', kind='pilot'),
                                   case, [], {}, {}, 'sdfampere036', [])

    def test_missing_pilots_and_diagnostics_block_full_phase(self):
        with self.assertRaisesRegex(ValueError, 'eight pilots'):
            launch.full_gate([])
        pilots = [(case, {'rows': [{}, {}] + [{'rank': rank, 'callbacks': [{}], 'timestamps': [rank]}
                                            for rank in range(2, case['bds'] + 2)]})
                  for case in plan.matrix()[:8]]
        with self.assertRaisesRegex(ValueError, 'diagnostic'):
            launch.full_gate(pilots)
        for case, result in pilots:
            for row in result['rows'][2:]:
                row['cuda_intervals'] = dict(interpretation='Instrumented CUDA intervals; never clean rates',
                    intervals=[dict(name=name, milliseconds=1.0) for name in
                               ('expansion', 'calibration', 'cube_count_and_sum')])
        launch.full_gate(pilots)

    def test_parent_shape_allows_zero_owned_acceptance_bds(self):
        case = plan.acceptance_cases()[1]
        result, stamps, cpu, gpu = fake_result(case, [1003, 0, 0, 0])
        launch.validate_result(result, case, stamps, cpu, gpu, 'sdfampere036', [])
        result['rows'][3]['io']['read_size_histogram'] = {'524724': 1}
        with self.assertRaisesRegex(ValueError, 'Read histogram'):
            launch.validate_result(result, case, stamps, cpu, gpu, 'sdfampere036', [])

    def test_zero_owned_diagnostics_do_not_waive_global_coverage(self):
        pilots = []
        for case in plan.matrix()[:8]:
            ownership = [case['events']] + [0] * (case['bds'] - 1)
            result, stamps, cpu, gpu = fake_result(case, ownership)
            for row in result['rows'][2:]:
                row['cuda_intervals'] = dict(interpretation='Separate pilot CUDA measurements',
                    intervals=[dict(name=name, milliseconds=1.0) for name in
                               ('expansion', 'calibration', 'cube_count_and_sum')] if row['callbacks'] else [])
            launch.validate_result(result, case, stamps, cpu, gpu, 'sdfampere036', [])
            pilots.append((case, result))
        launch.full_gate(pilots)
        case, result = pilots[-1]
        result['rows'][2]['timestamps'].pop()
        with self.assertRaisesRegex(ValueError, 'global disjoint identity'):
            launch.validate_result(result, case, stamps, cpu, gpu, 'sdfampere036', [])

    def test_no_submission_api_and_review_required_for_command(self):
        self.assertFalse(hasattr(prepare, 'submit'))
        with (patch.object(prepare, 'verify_manifest', side_effect=ValueError('review missing')),
              patch.object(prepare.subprocess, 'run') as run):
            with self.assertRaisesRegex(ValueError, 'review missing'):
                prepare.command(Path('/attempt'), Path('/review'))
            run.assert_not_called()

    def test_command_only_pins_node_and_paths(self):
        manifest = dict(node='sdfampere036', walltime_seconds=86400,
                        partition_evidence=dict(max_time_seconds=864000))
        with patch.object(prepare, 'verify_manifest', return_value=manifest), patch.object(prepare, 'sha', return_value='a' * 64):
            command = prepare.command(Path('/attempt'), Path('/review'))
        self.assertIn('--nodelist=sdfampere036', command)
        self.assertIn('--time=1440', command)
        self.assertEqual(command[-3:], ['/attempt', '/review', 'a' * 64])


if __name__ == '__main__':
    unittest.main()

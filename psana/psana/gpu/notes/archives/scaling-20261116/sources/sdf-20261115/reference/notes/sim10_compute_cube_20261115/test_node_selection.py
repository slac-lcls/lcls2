"""CPU-only allowlist regressions using in-memory evidence and no submissions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load('node_selection_plan_test', 'plan.py')
with patch.dict(sys.modules, {'plan': plan}):
    node_selection = load('node_selection_module_test', 'node_selection.py')
    with patch.dict(sys.modules, {'node_selection': node_selection}):
        prepare = load('node_selection_prepare_test', 'prepare.py')


def fake_allowlist():
    nodes = ['sdfampere020', 'sdfampere036']
    candidates, evidence = {}, {}
    nics = ['%s Ethernet controller [0200]: %s' % (address, model)
            for address, model in [('02:00.0', 'ConnectX-6'), ('42:00.0', 'ConnectX-6'),
                                   ('82:00.0', 'Ethernet Adapter'), ('82:00.1', 'Ethernet Adapter')]]
    for index, host in enumerate(nodes, 1):
        source = str(plan.SCRATCH / 'node-selection-unit-fixtures' / (host + '.json'))
        digest = str(index) * 64
        profile = dict(cpu=deepcopy(node_selection.CPU), gpu_numa=deepcopy(node_selection.GPU_NUMA),
                       driver='575.57.08', nics=nics[:],
                       gpu_uuids=['GPU-%s-%d' % (host, device) for device in range(4)])
        candidates[host] = dict(source=source, sha256=digest, profile=profile)
        evidence[source] = digest
    return dict(schema_version=1, partition='ampere', nodes=nodes,
                partition_nodes=['sdfampere019', *nodes, 'sdfampere037'],
                candidates=candidates, evidence_files=evidence)


def fake_hardware(record, host):
    profile = record['candidates'][host]['profile']
    cpu = json.dumps(dict(lscpu=[dict(field=field, data=value) for field, value in profile['cpu'].items()]))
    hardware = dict(host=host + '.example.test',
                    commands=dict(lscpu=dict(stdout=cpu),
                                  pci=dict(stdout='\n'.join(reversed(profile['nics']))),
                                  gpu=dict(stdout='Driver Version : ' + profile['driver'])),
                    pci_devices=[dict(bdf=address, numa_node=numa, max_link_speed='16.0 GT/s PCIe',
                                      max_link_width='16') for address, numa in profile['gpu_numa'].items()])
    devices = [dict(uuid=uuid) for uuid in reversed(profile['gpu_uuids'])]
    return hardware, devices


class CpuOnlyTests(unittest.TestCase):
    def setUp(self):
        process_patch = patch.object(subprocess, 'Popen', side_effect=AssertionError('External process forbidden'))
        self.process = process_patch.start()
        self.addCleanup(process_patch.stop)
        self.record = fake_allowlist()

    def tearDown(self):
        self.process.assert_not_called()


class NodeSelectionTests(CpuOnlyTests):
    def test_scheduler_excludes_complement_without_multinode_nodelist(self):
        original = deepcopy(self.record)
        live = list(reversed(self.record['partition_nodes'])) + [self.record['nodes'][0]]
        flags = node_selection.scheduler_flags(self.record, live)
        self.assertEqual(flags, ['--nodes=1', '--exclude=sdfampere019,sdfampere037'])
        self.assertFalse(any(flag.startswith('--nodelist') for flag in flags))
        self.assertEqual(self.record, original)

    def test_scheduler_rejects_stale_partition_membership(self):
        universe = self.record['partition_nodes']
        observations = [universe + ['sdfampere099'], universe[1:],
                        [host for host in universe if host != self.record['nodes'][0]]]
        for live in observations:
            with self.subTest(live=live), self.assertRaisesRegex(ValueError, 'Partition membership changed'):
                node_selection.scheduler_flags(self.record, live)

    def test_empty_invalid_duplicate_and_unsorted_allowlists(self):
        cases = [([], 'Nonempty canonical allowlist'),
                 (['not-a-node'], 'Invalid allowlist universe'),
                 (['sdfampere099'], 'Invalid allowlist universe'),
                 (['sdfampere020', 'sdfampere020', 'sdfampere036'], 'Nonempty canonical allowlist'),
                 (list(reversed(self.record['nodes'])), 'Nonempty canonical allowlist'),
                 (self.record['partition_nodes'][:], 'Invalid allowlist universe')]
        for names, message in cases:
            with self.subTest(nodes=names):
                record = deepcopy(self.record)
                record['nodes'] = names
                with self.assertRaisesRegex(ValueError, message):
                    node_selection.validate(record)
                with self.assertRaisesRegex(ValueError, message):
                    node_selection.scheduler_flags(record, record['partition_nodes'])

    def test_invalid_partition_universe_and_missing_evidence(self):
        for universe in ([], self.record['partition_nodes'] + ['wrong-node'],
                         self.record['partition_nodes'] + ['sdfampere037']):
            with self.subTest(universe=universe):
                record = deepcopy(self.record)
                record['partition_nodes'] = universe
                with self.assertRaises(ValueError):
                    node_selection.validate(record)
        record = deepcopy(self.record)
        record['candidates'].pop(record['nodes'][0])
        with self.assertRaisesRegex(ValueError, 'Candidate evidence coverage'):
            node_selection.validate(record)
        record = deepcopy(self.record)
        record['evidence_files'].clear()
        with self.assertRaisesRegex(ValueError, 'Missing inventory hash'):
            node_selection.validate(record)

    def test_selected_host_accepts_each_reviewed_candidate(self):
        manifest = dict(node=None, node_allowlist=self.record)
        for host in self.record['nodes']:
            with self.subTest(host=host):
                self.assertEqual(node_selection.selected_host(manifest, host, host), host)

    def test_selected_host_rejects_unreviewed_host(self):
        manifest = dict(node='sdfampere019', node_allowlist=self.record)
        for host in ('sdfampere019', 'sdfampere099'):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, 'outside reviewed node allowlist'):
                node_selection.selected_host(manifest, host, host)

    def test_selected_host_rejects_multinode_or_mismatched_allocation(self):
        manifest = dict(node=None, node_allowlist=self.record)
        host = self.record['nodes'][0]
        for allocation in (','.join(self.record['nodes']), 'sdfampere[020,036]',
                           self.record['nodes'][1], '', None):
            with self.subTest(allocation=allocation), self.assertRaisesRegex(ValueError, 'exactly the selected host'):
                node_selection.selected_host(manifest, host, allocation)

    def test_selected_host_preserves_legacy_singleton_guards(self):
        host = 'sdfampere036'
        for manifest in (dict(node=host), dict(node=host, node_allowlist=None)):
            with self.subTest(manifest=manifest):
                self.assertEqual(node_selection.selected_host(manifest, host, host), host)
                with self.assertRaisesRegex(ValueError, 'outside reviewed node allowlist'):
                    node_selection.selected_host(manifest, 'sdfampere020', 'sdfampere020')
                with self.assertRaisesRegex(ValueError, 'exactly the selected host'):
                    node_selection.selected_host(manifest, host, 'sdfampere020,sdfampere036')

    def test_hardware_gate_accepts_fake_fixture_for_each_selected_host(self):
        for host in self.record['nodes']:
            with self.subTest(host=host):
                hardware, devices = fake_hardware(self.record, host)
                result = node_selection.hardware_gate(self.record, host, hardware, devices)
                self.assertEqual(result['status'], 'pass')
                self.assertEqual(result['selected_node'], host)
                self.assertEqual(result['historical_inventory_sha256'], self.record['candidates'][host]['sha256'])

    def test_hardware_gate_rejects_cpu_drift(self):
        host = self.record['nodes'][0]
        for field in self.record['candidates'][host]['profile']['cpu']:
            with self.subTest(field=field):
                hardware, devices = fake_hardware(self.record, host)
                cpu = json.loads(hardware['commands']['lscpu']['stdout'])
                for row in cpu['lscpu']:
                    if row['field'] == field:
                        row['data'] = 'changed'
                hardware['commands']['lscpu']['stdout'] = json.dumps(cpu)
                with self.assertRaisesRegex(ValueError, 'Live CPU topology drift'):
                    node_selection.hardware_gate(self.record, host, hardware, devices)

    def test_hardware_gate_rejects_nic_drift(self):
        host = self.record['nodes'][0]
        hardware, devices = fake_hardware(self.record, host)
        hardware['commands']['pci']['stdout'] = hardware['commands']['pci']['stdout'].replace('ConnectX-6', 'ConnectX-7', 1)
        with self.assertRaisesRegex(ValueError, 'Live NIC inventory drift'):
            node_selection.hardware_gate(self.record, host, hardware, devices)

    def test_hardware_gate_rejects_gpu_uuid_drift(self):
        host = self.record['nodes'][0]
        hardware, devices = fake_hardware(self.record, host)
        devices[0]['uuid'] = 'GPU-unreviewed'
        with self.assertRaisesRegex(ValueError, 'Live GPU identity drift'):
            node_selection.hardware_gate(self.record, host, hardware, devices)

    def test_hardware_gate_rejects_pci_numa_and_link_drift(self):
        host = self.record['nodes'][0]
        for address in self.record['candidates'][host]['profile']['gpu_numa']:
            for field, value in (('numa_node', '-1'), ('max_link_speed', '8.0 GT/s PCIe'), ('max_link_width', '8')):
                with self.subTest(address=address, field=field):
                    hardware, devices = fake_hardware(self.record, host)
                    endpoint = next(row for row in hardware['pci_devices'] if row['bdf'] == address)
                    endpoint[field] = value
                    with self.assertRaisesRegex(ValueError, 'Live GPU PCI/NUMA drift'):
                        node_selection.hardware_gate(self.record, host, hardware, devices)

    def test_hardware_gate_rejects_changed_or_missing_driver(self):
        host = self.record['nodes'][0]
        for text in ('Driver Version : 580.00.00', ''):
            with self.subTest(driver=text):
                hardware, devices = fake_hardware(self.record, host)
                hardware['commands']['gpu']['stdout'] = text
                with self.assertRaisesRegex(ValueError, 'Live driver drift'):
                    node_selection.hardware_gate(self.record, host, hardware, devices)

    def test_hardware_gate_rejects_other_hosts_inventory(self):
        host, other = self.record['nodes']
        hardware, devices = fake_hardware(self.record, other)
        with self.assertRaisesRegex(ValueError, 'Hardware host mismatch'):
            node_selection.hardware_gate(self.record, host, hardware, devices)


class PrepareCommandTests(CpuOnlyTests):
    def setUp(self):
        super().setUp()
        self.attempt = plan.SCRATCH / 'node-selection-unit-fixtures' / 'attempt'
        self.review = self.attempt / 'review.json'
        self.manifest = dict(node=None, node_allowlist=self.record, walltime_seconds=61,
                             partition_evidence=dict(max_time_seconds=3600))
        self.live_nodes = list(reversed(self.record['partition_nodes']))
        patches = dict(verify=patch.object(prepare, 'verify_manifest', return_value=self.manifest),
                       digest=patch.object(prepare, 'sha', return_value='a' * 64),
                       discovery=patch.object(prepare.subprocess, 'check_output', side_effect=self.live_membership),
                       collect=patch.object(node_selection, 'collect', side_effect=AssertionError('Do not replace reviewed evidence')))
        for name, patcher in patches.items():
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def live_membership(self, command, **kwargs):
        self.verify.assert_called_once_with(self.attempt, self.review)
        self.assertEqual(command, ['sinfo', '-p', 'ampere', '-N', '-h', '-o', '%N'])
        self.assertEqual(kwargs, dict(text=True, timeout=30))
        return '\n'.join(self.live_nodes)

    def tearDown(self):
        self.collect.assert_not_called()
        super().tearDown()

    def test_command_uses_reviewed_allowlist_and_live_membership_without_submission(self):
        with patch.object(node_selection, 'scheduler_flags', wraps=node_selection.scheduler_flags) as selection:
            command = prepare.command(self.attempt, self.review)
        self.verify.assert_called_once_with(self.attempt, self.review)
        self.discovery.assert_called_once()
        selection.assert_called_once_with(self.record, self.live_nodes)
        self.digest.assert_called_once_with(self.review)
        self.assertEqual(command, ['sbatch', '--parsable', '--nodes=1', '--exclude=sdfampere019,sdfampere037',
            '--time=2', '--chdir=' + str(self.attempt), '--output=' + str(self.attempt / 'slurm-%j.out'),
            '--error=' + str(self.attempt / 'slurm-%j.err'),
            str(self.attempt / 'source/notes' / plan.HERE.name / 'run.sbatch'),
            str(self.attempt), str(self.review), 'a' * 64])
        self.assertFalse(any(flag.startswith('--nodelist') for flag in command))

    def test_command_rejects_stale_live_partition(self):
        self.live_nodes.append('sdfampere099')
        with self.assertRaisesRegex(ValueError, 'Partition membership changed'):
            prepare.command(self.attempt, self.review)
        self.discovery.assert_called_once()
        self.digest.assert_not_called()

    def test_command_requires_review_before_live_discovery(self):
        self.verify.side_effect = ValueError('review missing')
        with self.assertRaisesRegex(ValueError, 'review missing'):
            prepare.command(self.attempt, self.review)
        self.verify.assert_called_once_with(self.attempt, self.review)
        self.discovery.assert_not_called()
        self.digest.assert_not_called()

    def test_command_preserves_legacy_singleton_without_discovery(self):
        self.manifest.pop('node_allowlist')
        self.manifest['node'] = 'sdfampere036'
        command = prepare.command(self.attempt, self.review)
        self.verify.assert_called_once_with(self.attempt, self.review)
        self.discovery.assert_not_called()
        self.assertEqual([flag for flag in command if flag.startswith('--nodelist')], ['--nodelist=sdfampere036'])
        self.assertFalse(any(flag.startswith('--exclude') for flag in command))


if __name__ == '__main__':
    unittest.main()

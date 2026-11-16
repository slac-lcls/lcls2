import unittest
from unittest.mock import Mock, patch

import numpy as np

from contracts import admission, admission_from_budget, memory_demand
from reference import CpuCalibration, CubeReference, samples


class ContractsTest(unittest.TestCase):
    def test_budget_methods_are_called(self):
        resident = 1024
        allocations = [dict(capacity=resident, category='test')]
        budget = Mock()
        budget.committed.return_value = resident
        budget.allocation_snapshot.return_value = allocations
        needed = sum(memory_demand('S', 50).values()) - resident
        result = admission_from_budget('S', 50, needed, budget)
        self.assertEqual(result['framework_resident'], resident)
        self.assertEqual(result['framework_allocations'], allocations)
        budget.committed.assert_called_once_with()
        budget.allocation_snapshot.assert_called_once_with()

    def test_exact_memory_subtotals(self):
        for batch, gib in ((50, 16.9375), (200, 31.0)):
            demand = memory_demand('ECC', batch)
            self.assertEqual(demand['images'] + demand['cube'] + demand['framework_ceiling'],
                             int(gib * (1 << 30)) + 800)
        self.assertEqual(memory_demand('S', 200)['images'], 0)

    def test_resident_constants_not_double_charged(self):
        resident = 1 << 30
        needed = sum(memory_demand('ECC', 200).values()) - resident
        allocations = [dict(capacity=resident, category='task-constants')]
        receipt = admission('ECC', 200, needed, resident, allocations)
        self.assertEqual(receipt['outstanding'], needed)
        with self.assertRaises(ValueError):
            admission('ECC', 200, needed - 1, resident, allocations)
        with self.assertRaises(ValueError):
            admission('ECC', 200, needed, resident, [])

    def test_reference_constant_packing(self):
        pedestal = np.full((3, 2, 3), 0.1, np.float64)
        gain = np.full_like(pedestal, 3)
        offset = np.full_like(pedestal, 0.2)
        gain[1, 0, 0] = 0
        constants = {key: (value, {}) for key, value in
                     [('pedestals', pedestal), ('pixel_gain', gain), ('pixel_offset', offset)]}
        reference = CpuCalibration(constants)
        self.assertEqual(reference.constants.dtype, np.float32)
        self.assertTrue(np.all(reference.constants[2] == 0))
        self.assertTrue(np.all(reference.constants[0, :, 0] == np.float32(0.1 + 0.2)))
        self.assertEqual(reference.constants[1, 0, 1], 0)
        self.assertEqual(reference.constants[3, 0, 1], np.float32(1 / 3))

    def test_samples_cover_edges_and_repeat(self):
        selected = samples()
        self.assertTrue(np.array_equal(selected, samples()))
        self.assertTrue({0, 9, 10, 512 * 1024 - 1}.issubset(set(selected)))
        self.assertEqual(len(selected), len(set(selected)))

    def test_reference_rejects_wrong_count_and_empty_bin(self):
        class Reader:
            timestamps = list(range(200))
            def raw(self, timestamp, selected):
                return np.array([timestamp], np.float32)

        class Calibration:
            shape = (1,)
            def __init__(self, constants, selected):
                pass
            def __call__(self, raw):
                return raw

        with patch('reference.CpuCalibration', Calibration):
            reference = CubeReference(Reader(), {}, True)
        reference(0, np.array([297.0]), 3)
        reference(99, np.zeros(1), 0)
        with self.assertRaises(ValueError):
            reference(0, np.array([297.0]), 2)
        with self.assertRaises(ValueError):
            reference(99, np.ones(1), 0)


if __name__ == '__main__':
    unittest.main()

import struct
import unittest

from common import cases, check_rows, pair_order, scan_smd


def dgram(service, timestamp):
    return struct.pack('<IIIIHHI', timestamp, 1, service << 24, 0, 0, 0, 12)


class ReferenceTests(unittest.TestCase):
    def test_scan(self):
        payload = dgram(6, 0) + dgram(12, 1) + dgram(7, 2) + dgram(6, 3) + dgram(12, 4) + dgram(5, 5)
        events, ends = scan_smd(payload)
        self.assertEqual(events, [(2**32 + 1, 0), (2**32 + 4, 1)])
        self.assertEqual(ends, 1)

    def test_invalid(self):
        for payload in (b'x', dgram(12, 1), dgram(6, 0) + dgram(12, 1) * 2,
                        dgram(6, 0) + dgram(12, 3) + dgram(12, 2)):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                scan_smd(payload)

    def test_matrix(self):
        matrix = cases()
        self.assertEqual(len(matrix), 8)
        self.assertEqual(sum(len(case['variants']) * 4 for case in matrix), 64)
        for case in matrix:
            if case['run'] == 77:
                self.assertEqual({variant['runtime'] for variant in case['variants']}, {'head'})
        self.assertEqual([pair_order(index) for index in range(4)], [(0, 1), (1, 0), (0, 1), (1, 0)])

    def test_identities(self):
        rows = [dict(role='smd0', timestamps=[], step_records=[]),
                dict(role='bd', timestamps=[1, 3], step_records=[(1, 0), (3, 1)]),
                dict(role='bd', timestamps=[2], step_records=[(2, 0)])]
        expected = [(1, 0), (2, 0), (3, 1)]
        self.assertTrue(check_rows(rows, expected, 'steps'))
        rows[2]['timestamps'] = [3]
        with self.assertRaises(ValueError):
            check_rows(rows, expected, 'events')


if __name__ == '__main__':
    unittest.main()

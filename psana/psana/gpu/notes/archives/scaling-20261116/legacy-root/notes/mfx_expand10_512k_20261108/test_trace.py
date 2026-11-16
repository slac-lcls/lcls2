import unittest
from analyze_trace import intervals


class Tests(unittest.TestCase):
    def test_overlap_counts_distinct_streams_not_nested_spans(self):
        got=intervals([(0,10,1),(2,8,1),(5,15,2),(7,12,3)])
        self.assertEqual(got['kernel_union_ns'],15)
        self.assertEqual(got['cross_stream_overlap_ns'],7)
        self.assertEqual(got['max_concurrent_streams'],3)

    def test_touching_intervals_and_idle_gaps_are_not_overlap(self):
        got=intervals([(0,5,1),(5,10,2),(12,14,1)])
        self.assertEqual(got['kernel_union_ns'],12)
        self.assertEqual(got['cross_stream_overlap_ns'],0)
        self.assertEqual(got['max_concurrent_streams'],1)

    def test_empty_or_invalid_trace_is_not_evidence_of_no_overlap(self):
        for rows in ([],[(2,2,1)],[(3,2,1)]):
            with self.assertRaises(ValueError):intervals(rows)


if __name__=='__main__':unittest.main()

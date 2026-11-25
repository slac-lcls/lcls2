import unittest
from sequence import ordered_groups


class SubmissionOrdering(unittest.TestCase):
    def test_failed_predecessor_never_launches_successor(self):
        calls=[]
        def launch(p):calls.append(('launch',p))
        def check(p):
            calls.append(('check',p))
            if p==2:raise ValueError('Numerical audit failed')
            return p
        with self.assertRaises(ValueError):ordered_groups([1,2,4],launch,check)
        self.assertEqual(calls,[('launch',1),('check',1),('launch',2),('check',2)])

    def test_submission_failure_never_checks_or_advances(self):
        calls=[]
        def launch(p):calls.append(p);raise RuntimeError('Allocation failed')
        self.assertRaises(RuntimeError,ordered_groups,[1,2,4],launch,lambda p:self.fail('Cannot audit an unlaunched group'))
        self.assertEqual(calls,[1])


if __name__=='__main__':unittest.main()

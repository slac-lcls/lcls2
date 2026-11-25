import unittest
from sequence import granted_job,terminal_accounting
class SequenceChecks(unittest.TestCase):
 def test_failed_or_ambiguous_launch_never_passes(self):
  group={'name':'paired-b01-bd1','attempt':'/attempt'}
  for rc,console in [(1,'Granted job allocation 123'),(0,''),(0,'Granted job allocation 123\nGranted job allocation 124')]:
   with self.assertRaises(ValueError):granted_job(dict(group,returncode=rc),group,console)
  self.assertEqual(granted_job(dict(group,returncode=0),group,'Granted job allocation 123'),'123')
 def test_pending_or_missing_allocation_never_terminal(self):
  for s in ('','123.0|COMPLETED|0:0|nid1','123|RUNNING|0:0|nid1','123|COMPLETED|0:0|nid1\n123.0|COMPLETING|0:0|nid1'):
   self.assertFalse(terminal_accounting('123',s))
  # Failed terminal jobs reach the reviewed completed() gate, which rejects them.
  self.assertTrue(terminal_accounting('123','123|FAILED|1:0|nid1'))
if __name__=='__main__':unittest.main()

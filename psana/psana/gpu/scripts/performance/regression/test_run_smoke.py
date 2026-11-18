import copy
import unittest

from run_smoke import derive_prefix


class SmokePrefixTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(schema='run51-raw-reference-v1', events=[[1,0,120],[2,0,120],[3,0,120]])
        self.plan = dict(schema='run51-input-plan-v1', reference=self.reference, streams=[])
        self.manifest = dict(schema='run51-private-input-v2', private_copy=True, files=[])
        for stream in (3,5,7,8,9):
            self.plan['streams'].append(dict(stream=stream, smd=dict(sha256='smd'),
                 events=[[1,0,0,24,12],[2,0,48,24,11],[3,0,72,24,12]]))
            self.manifest['files'].append(dict(stream=stream, smd=dict(sha256='smd'),
                 bigdata=dict(size=96,ranges=[[0,24],[48,48]],range_bytes=72,sha256='parent')))

    def test_prefix_excludes_transition_gap_and_drops_parent_hash(self):
        original = copy.deepcopy(self.manifest)
        manifest, reference = derive_prefix(self.manifest,self.plan,self.reference,2)
        self.assertEqual(reference['events'],self.reference['events'][:2])
        self.assertEqual(manifest['files'][0]['bigdata']['ranges'],[[0,24],[48,24]])
        self.assertEqual(sum(f['bigdata']['range_bytes'] for f in manifest['files']),240)
        self.assertNotIn('sha256',manifest['files'][0]['bigdata'])
        self.assertEqual(self.manifest,original)

    def test_rejects_inconsistent_parent_ranges_reference_and_streams(self):
        for kind in ('range','reference','identity','smd','stream'):
            manifest, plan, reference = copy.deepcopy((self.manifest,self.plan,self.reference))
            if kind == 'range': manifest['files'][0]['bigdata']['ranges']=[[0,96]]
            if kind == 'reference': reference['events'][0][2]=121
            if kind == 'identity': plan['streams'][1]['events'][0][0]=9
            if kind == 'smd': manifest['files'][0]['smd']['sha256']='changed'
            if kind == 'stream': manifest['files'][0]['stream']=4
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                derive_prefix(manifest,plan,reference,2)

    def test_rejects_invalid_prefix_length(self):
        for count in (0,4,True,1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                derive_prefix(self.manifest,self.plan,self.reference,count)


if __name__ == '__main__':
    unittest.main()

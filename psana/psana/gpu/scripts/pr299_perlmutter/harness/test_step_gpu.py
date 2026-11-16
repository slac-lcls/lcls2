import unittest
from unittest.mock import patch
import gpu_identity as identity

UUID='GPU-c66a5240-c553-4f3b-8faf-888aadef5938'
OTHER='GPU-11111111-2222-3333-4444-555555555555'
class StepGpuTests(unittest.TestCase):
    def env(self,**changes):
        return dict(dict(SLURM_JOB_ID='1',SLURM_JOB_GPUS='0,1,2,3',SLURM_STEP_GPUS='2',CUDA_VISIBLE_DEVICES='0'),**changes)
    def test_whole_node_job_with_single_gpu_step(self):
        identity.validate_environment(self.env())
    def test_multi_gpu_step_and_visibility_rejected(self):
        for changes in [dict(SLURM_STEP_GPUS='0,1'),dict(CUDA_VISIBLE_DEVICES='0,1'),dict(SLURM_STEP_GPUS='4'),dict(SLURM_STEP_GPUS=None)]:
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):identity.validate_environment(self.env(**changes))
    def test_step_preserves_slurm_visibility(self):
        env=self.env()
        with patch.dict(identity.os.environ,env,clear=True),patch.object(identity,'cuda_identity',return_value=dict(uuid=UUID,count=1)):
            record=identity.verify_step_gpu(UUID)
            self.assertEqual(identity.os.environ['CUDA_VISIBLE_DEVICES'],'0')
            self.assertEqual(record['environment'],env)
    def test_wrong_physical_gpu_rejected(self):
        with patch.dict(identity.os.environ,self.env(),clear=True),patch.object(identity,'cuda_identity',return_value=dict(uuid=OTHER,count=1)):
            with self.assertRaisesRegex(RuntimeError,'UUID differs'):identity.verify_step_gpu(UUID)
    def test_job_without_step_rejected_on_bd(self):
        with patch.dict(identity.os.environ,self.env(SLURM_JOB_GPUS='0',SLURM_STEP_GPUS=''),clear=True):
            with self.assertRaisesRegex(RuntimeError,'explicit single-GPU'):identity.verify_step_gpu(UUID)
if __name__=='__main__':unittest.main()

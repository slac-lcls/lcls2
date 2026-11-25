import unittest
from pathlib import Path
from launch_steps import step_command,step_environment
from topology import rank_hosts


class SlurmLaunch(unittest.TestCase):
    def test_ranked_steps_preserve_asymmetric_rank_layout(self):
        for nodes in (1,2,4):
            hosts=['nid%06d'%n for n in range(nodes)]
            argv=step_command(Path('/bundle'),Path('/attempt'),hosts,'probe.py',[],120,True)
            self.assertFalse(any(a.startswith(('--nodes=','--nodelist=')) for a in argv))
            self.assertIn('--distribution=arbitrary',argv)
            self.assertIn('--ntasks='+str(4*nodes+2),argv)
            layout=rank_hosts(hosts)
            self.assertEqual(layout.count(hosts[0]),6)
            for host in hosts[1:]:self.assertEqual(layout.count(host),4)

    def test_parent_layout_does_not_override_rank_hostfile(self):
        parent={k:'stale' for k in ('SLURM_HOSTFILE','SLURM_NTASKS_PER_NODE','SLURM_TASKS_PER_NODE','SLURM_DISTRIBUTION','SLURM_NNODES','SLURM_JOB_NUM_NODES','SLURM_NODELIST','SLURM_NTASKS','SLURM_NPROCS')}
        parent.update(SLURM_JOB_ID='123',SLURM_JOB_NODELIST='nid[000001-000004]',CUDA_VISIBLE_DEVICES='0,1,2,3')
        env=step_environment(parent,Path('/correct/hosts'))
        self.assertEqual(env,dict(SLURM_HOSTFILE='/correct/hosts',SLURM_JOB_ID='123',SLURM_JOB_NODELIST='nid[000001-000004]',CUDA_VISIBLE_DEVICES='0,1,2,3'))
        self.assertEqual(parent['SLURM_HOSTFILE'],'stale')

    def test_per_node_steps_clear_hostfile_and_select_each_node(self):
        hosts=['nid000001','nid000002']
        argv=step_command(Path('/bundle'),Path('/attempt'),hosts,'probe.py',[],120)
        for flag in ('--nodes=2','--ntasks=2','--nodelist=nid000001,nid000002','--distribution=block'):self.assertIn(flag,argv)
        self.assertNotIn('SLURM_HOSTFILE',step_environment({'SLURM_HOSTFILE':'stale'}))

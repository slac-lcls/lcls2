import os, shutil
import subprocess

# cpo and weninc split off this test because of an issue with openmpi where
# a python file that does "import mpi4py" cannot fork an "mpirun" command.
# see: https://bitbucket.org/mpi4py/mpi4py/issues/95/mpi4py-openmpi-300-breaks-subprocess

class Test:
    def setup_input_files(sefl, tmp_path):
        xtc_dir = tmp_path / '.tmp'
        xtc_dir.mkdir()
        smd_dir = xtc_dir / 'smalldata'
        smd_dir.mkdir()

        # segments 0,1 and "counting" timestamps for event-building
        s01file = str(xtc_dir / 'data-r0001-s00.xtc2')
        subprocess.call(['xtcwriter','-f',s01file,'-t'])
        subprocess.call(['smdwriter','-f',s01file,'-o',str(smd_dir / 'data-r0001-s00.smd.xtc2')])

        # segments 2,3
        s23file = str(xtc_dir / 'data-r0001-s01.xtc2')
        subprocess.call(['xtcwriter','-f',s23file,'-t','-s','2',])
        subprocess.call(['smdwriter','-f',s23file,'-o',str(smd_dir / 'data-r0001-s01.smd.xtc2')])

        subprocess.call(['epicswriter','-f',s01file,'-o',str(xtc_dir / 'data-r0001-s02.xtc2')])
        # Epics data with streamId 2
        subprocess.call(['epicswriter','-f',s01file,'-o',str(xtc_dir / 'data-r0001-s03.xtc2'), '-s', '2'])

        shutil.copy(s01file,str(smd_dir / 'data-r0002-s00.smd.xtc2'))
        shutil.copy(s23file,str(smd_dir / 'data-r0002-s01.smd.xtc2'))

    # as a result of the problem described above, this test assumes all
    # input files have been setup by test_xtc.py
    def test_mpi(self, tmp_path):
        self.setup_input_files(tmp_path)

        env = dict(list(os.environ.items()) + [
            ('TEST_XTC_DIR', str(tmp_path)),
        ])

        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_based], env=env)

        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        subprocess.check_call(['mpirun','-n','3','python',callback_based], env=env)
        
        loop_exhaustive_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_exhaustive_based], env=env)

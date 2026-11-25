"""Launch the unchanged PR326 driver with independent synthetic fixtures."""
import os,sys
from pathlib import Path
import pytest
TESTS=Path(__file__).resolve().parent/'source/psana/psana/tests'
sys.path.insert(0,str(TESTS))
from test_callback_chunk_state import run_process

@pytest.mark.parametrize('ebs,nranks', [(1,4),(2,5)])
def test_pr326_multirun(tmp_path,ebs,nranks):
    xtc=tmp_path/'.tmp';smd=xtc/'smalldata';smd.mkdir(parents=True)
    env=os.environ.copy()
    for stream in range(2):
        stem=f'xpptut15-r0014-s{stream:03d}-c000';f=xtc/(stem+'.xtc2')
        run_process(['xtcwriter','-f',str(f),'-t','-n','10','-s',str(stream*2),'-e','4','-m','1'],env)
        run_process(['smdwriter','-f',str(f),'-o',str(smd/(stem+'.smd.xtc2'))],env)
    env.update(PS_PARALLEL='mpi',PS_EB_NODES=str(ebs),TEST_XTC_DIR=str(tmp_path))
    output=run_process(['mpirun','-n',str(nranks),sys.executable,'-B',str(TESTS/'run_multirun.py')],env)
    (tmp_path/'driver.log').write_text(output)
    assert 'events per run over all ranks: [10, 10]' in output

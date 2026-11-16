"""CPU-only port tests under the exact native runtime; no Slurm/GPU launches."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from support import HERE, PREFIX, require, sha, write

os.environ['PS_PARALLEL']='none'
os.environ['CUDA_VISIBLE_DEVICES']=''
output=Path(sys.argv[1]);output.mkdir(exist_ok=True)
for p in HERE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
for name in ('runtime.sh','run.sbatch'):subprocess.run(['bash','-n',str(HERE/name)],check=True)
import psana
require(Path(psana.__file__).resolve().is_relative_to(PREFIX),'Wrong installed psana')
sys.path.insert(1,str(HERE/'vendor/sim10_compute_cube_20261115'))
import stages,runner,device_check
paths=[HERE/'test_port.py',HERE/'test_audit_port.py',HERE/'test_boundaries.py',HERE/'test_cube_check.py']+[HERE/'vendor/sim10_compute_cube_20261115'/name for name in
       ('test_reference.py','test_mpi_cube.py','test_contracts.py')]
command=[sys.executable,'-m','pytest','-q','--basetemp='+str(output/'pytest-tmp'),*map(str,paths)]
r=subprocess.run(command,capture_output=True,text=True)
(output/'pytest.log').write_text(r.stdout+r.stderr)
commit=subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip()
working_status=subprocess.check_output(['git','-C',str(HERE),'status','--porcelain','--',str(HERE)],text=True)
write(output/'checks.json',dict(git_head=commit,task_working_status=working_status,status='pass' if r.returncode==0 else 'fail',exitcode=r.returncode,
      command=command,psana=psana.__file__,source_hashes={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.sbatch','.json')
          and not {'evidence','review','__pycache__'}.intersection(p.relative_to(HERE).parts)},
      limitations=['CPU-only; no GPU acceptance or scheduler execution. Host checks ran outside the unavailable Codex sandbox.']))
print(r.stdout[-3000:]);print(r.stderr[-1000:]);sys.exit(r.returncode)

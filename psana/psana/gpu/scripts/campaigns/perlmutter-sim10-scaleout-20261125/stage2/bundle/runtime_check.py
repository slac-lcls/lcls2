"""Selected installed runtime integrity, without GPU work or payload hashing."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selected():
 role=os.environ.get('SIM10_RUNTIME')
 if role not in ('P','C'):raise ValueError('Unknown runtime')
 return json.loads((HERE/'runtimes.json').read_text())[role]
def check():
 r=selected();source=Path(r['source']);prefix=Path(r['prefix'])
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==r['commit']
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
 assert sha(r['build_receipt'])==r['build_receipt_sha256']
 for relative,digest in r['installed'].items():assert sha(prefix/relative)==digest,relative
 for relative,digest in r['tests'].items():assert sha(source/relative)==digest,relative
 import psana
 assert Path(psana.__file__).resolve().is_relative_to(prefix)
 from support import input_info,verify_inputs
 info=input_info();verify_inputs(info)
 from task_runtime import kvikio_identity
 import importlib.util
 cupy=Path(importlib.util.find_spec('cupy').origin).parent
 identity={str(p):sha(p) for p in sorted(cupy.rglob('*')) if p.is_file() and p.suffix in ('.py','.so')}
 expected=json.loads((HERE/'dependency-pins.json').read_text())
 assert identity==expected['cupy_identity'] and kvikio_identity()==expected['kvikio_identity']
 result=dict(status='pass',role=r['role'],commit=r['commit'],prefix=str(prefix),counts=dict(installed=len(r['installed']),tests=len(r['tests'])),kvikio_identity=expected['kvikio_identity'],cupy_identity=identity,checked_unix=time.time())
 Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('kvikio_identity','cupy_identity')}))
if __name__=='__main__':check()

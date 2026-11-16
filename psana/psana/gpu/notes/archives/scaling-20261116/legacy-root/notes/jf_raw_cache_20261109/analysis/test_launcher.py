"""Exercise the actual launcher's EXIT/TERM cleanup with a harmless child."""
import json,os,shlex,subprocess,sys,tempfile,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
class Launcher(unittest.TestCase):
    def run_mode(self,mode):
        with tempfile.TemporaryDirectory(prefix='jf-cleanup-test-') as t:
            root=Path(t);out=root/'output';fake=root/'child.py';script=root/'run.sh'
            fake.write_text("import os,sys,time\nfrom pathlib import Path\np=Path(sys.argv[1]);p.mkdir();(p/'child').write_text(str(os.getpid()))\nif sys.argv[2]=='term':time.sleep(60)\n")
            source=(HERE/'run.sbatch').read_text();lines=source.splitlines()
            selected=[i for i,line in enumerate(lines) if line.startswith('bash "$source_dir/runtime.sh"')]
            self.assertEqual(len(selected),1)
            lines[selected[0]]=shlex.quote(sys.executable)+' '+shlex.quote(str(fake))+' "$output" '+mode+' &'
            script.write_text('\n'.join(lines)+'\n')
            env=dict(os.environ,SLURM_TMPDIR=t,SLURM_JOB_ID='12345')
            p=subprocess.Popen(['bash',str(script),str(HERE),str(out)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                if mode=='term':
                    for _ in range(200):
                        if (out/'child').exists():break
                        if p.poll() is not None:break
                        time.sleep(.025)
                    self.assertTrue((out/'child').exists());p.terminate()
                log,_=p.communicate(timeout=10)
                self.assertEqual(p.returncode,143 if mode=='term' else 0,log)
                r=json.loads((out/'root-cleanup.json').read_text());self.assertTrue(r['removed']);self.assertFalse((root/'jf-raw-cache-12345').exists())
                if mode=='term':
                    with self.assertRaises(ProcessLookupError):os.kill(int((out/'child').read_text()),0)
            finally:
                if p.poll() is None:p.kill();p.wait()
    def test_normal_exit(self):self.run_mode('normal')
    def test_term(self):self.run_mode('term')

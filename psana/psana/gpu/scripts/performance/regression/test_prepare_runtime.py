"""Runtime provenance/failure tests; tiny fake build, no CUDA or experiment data."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import native_contract as n
import prepare_runtime as p

BUILDER = '''#!/bin/bash
set -eu
while [ $# -gt 0 ]; do
 case "$1" in
 --prefix) prefix="$2";shift 2;;
 --build-dir) build="$2";shift 2;;
 *) shift;;
 esac
done
"$PYTHON" - "$prefix" "$build" <<'PY'
from pathlib import Path
import sys,json
prefix,build=map(Path,sys.argv[1:]);site=prefix/('lib/python%d.%d/site-packages'%sys.version_info[:2]);pkg=site/'psana'
pkg.mkdir(parents=True);(pkg/'__init__.py').write_text('');(pkg/'dgram.py').write_text('')
(prefix/'lib/support.so').write_bytes(b'fixture')
(build/'meson-info').mkdir(parents=True)
(build/'meson-info/intro-buildoptions.json').write_text(json.dumps([dict(name='prefix',value=str(prefix))]))
print('fixture build completed')
PY
'''


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.repo=self.root/'repo';self.repo.mkdir()
        self.git('init','-q');self.git('config','user.email','fixture@example.invalid');self.git('config','user.name','Fixture')
        (self.repo/'build_psana.sh').write_text(BUILDER)
        self.git('add','.');self.git('commit','-qm','fixture')
        self.sha=self.git('rev-parse','HEAD')
        self.cuda=self.root/'cuda';self.cuda.mkdir()
        self.env=dict(PATH=str(Path(sys.executable).parent)+':'+os.defpath,LD_LIBRARY_PATH=str(self.cuda),CUDA_HOME=str(self.cuda),CUDA_PATH=str(self.cuda),CONDA_PREFIX=str(Path(sys.executable).resolve().parent.parent),LCLS_CALIB_HTTP='https://example.invalid/calib')
        self.envfile=self.root/'environment.json';self.envfile.write_text(json.dumps(self.env))
        self.args=argparse.Namespace(repo=self.repo,commit=self.sha,root=self.root/'attempt',python=Path(sys.executable),environment=self.envfile,jobs=1,build_type='debugoptimized')

    def tearDown(self):self.tmp.cleanup()
    def git(self,*args):return subprocess.check_output(['git','-C',str(self.repo),*args],text=True).strip()
    def build(self):
        p.build(self.args)
        path=self.args.root/'build-complete.json'
        return path,p.file_hash(path)

    def test_real_detached_build_and_tamper_rejected(self):
        path,sha=self.build();r=p.checked_build(path,sha)
        self.assertEqual(r['commit'],self.sha);self.assertEqual(p.git(r['source'],'branch','--show-current'),'')
        (Path(r['install'])/'lib/support.so').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'installed files changed'):p.checked_build(path,sha)

    def test_dirty_original_is_not_used_and_source_changes_fail(self):
        (self.repo/'untracked.txt').write_text('unrelated')
        path,sha=self.build();r=p.checked_build(path,sha)
        self.assertFalse((Path(r['source'])/'untracked.txt').exists())
        (Path(r['source'])/'build_psana.sh').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source changed'):p.checked_build(path,sha)

    def test_no_overwrite_or_short_commit(self):
        self.args.commit=self.sha[:9]
        with self.assertRaisesRegex(ValueError,'full commit'):p.build(self.args)
        self.assertFalse(self.args.root.exists())
        self.args.commit=self.sha;self.build()
        with self.assertRaisesRegex(ValueError,'new build root'):p.build(self.args)

    def test_failed_build_preserved_without_completion(self):
        (self.repo/'build_psana.sh').write_text('#!/bin/bash\nexit 7\n');self.git('commit','-qam','failure');self.args.commit=self.git('rev-parse','HEAD')
        with self.assertRaises(subprocess.CalledProcessError):p.build(self.args)
        self.assertTrue((self.args.root/'build-failed.json').exists())
        self.assertFalse((self.args.root/'build-complete.json').exists())

    def test_receipt_hash_and_extra_file_rejected(self):
        path,sha=self.build()
        with self.assertRaisesRegex(ValueError,'receipt changed'):p.checked_build(path,'0'*64)
        r=p.read(path);(Path(r['install'])/'extra').write_text('unexpected')
        with self.assertRaisesRegex(ValueError,'installed files changed'):p.checked_build(path,sha)

    def test_external_symlink_and_directory_link_rejected(self):
        folder=self.root/'install';folder.mkdir();(folder/'code.py').write_text('pass')
        (folder/'external.py').symlink_to(self.envfile)
        with self.assertRaisesRegex(ValueError,'external'):p.inventory(folder)
        (folder/'external.py').unlink();(folder/'dir').symlink_to(self.cuda)
        with self.assertRaisesRegex(ValueError,'directory symlink'):p.inventory(folder)

    def test_environment_and_path_isolation(self):
        runtime=dict(environment=self.env,install='/selected',pythonpath='/selected/site',path_policy='isolated-install-v1')
        with patch.dict(os.environ,dict(PYTHONPATH='/other',LD_PRELOAD='/other.so',CUDA_VISIBLE_DEVICES='0')):
            env=n.native_environment(dict(contract=dict(environment={})),runtime)
        self.assertEqual(env['PATH'],'/selected/bin:'+self.env['PATH'])
        self.assertEqual(env['LD_LIBRARY_PATH'],'/selected/lib:'+self.env['LD_LIBRARY_PATH'])
        self.assertEqual(env['PYTHONPATH'],'/selected/site');self.assertNotIn('LD_PRELOAD',env)
        runtime['path_policy']='unknown'
        with self.assertRaisesRegex(ValueError,'path policy'):n.native_environment(dict(contract=dict(environment={})),runtime)
        self.env['PATH']=':'+self.env['PATH'];self.envfile.write_text(json.dumps(self.env))
        with self.assertRaisesRegex(ValueError,'path entries'):p.shared_environment(self.envfile,sys.executable)

    def test_pair_rejects_different_dependencies_or_reused_prefix(self):
        r={k:{} for k in p.SHARED};r.update(schema='psana-native-runtime-v1',path_policy='isolated-install-v1',api_base='199575205',files={'x':'a'},install_files={'x':'a'},commit='a'*40,source='/source-a',install='/a',package='/a/site/psana',pythonpath='/a/site')
        p.check_pair(r,copy.deepcopy(r))
        other=copy.deepcopy(r);other['dependencies']={'bad':'hash'}
        with self.assertRaisesRegex(ValueError,'dependencies'):p.check_pair(r,other)
        other=copy.deepcopy(r);other['commit']='b'*40
        with self.assertRaisesRegex(ValueError,'share an install'):p.check_pair(r,other)
        other.update(install='/b',package='/b/site/psana',pythonpath='/b/site',source='/source-b');p.check_pair(r,other)
        other['install']='/a/nested'
        with self.assertRaisesRegex(ValueError,'nested'):p.check_pair(r,other)
        other=copy.deepcopy(r);other['files']={'changed':'x'}
        with self.assertRaisesRegex(ValueError,'same captured manifest'):p.check_pair(r,other)

    def test_capture_failure_keeps_log_without_manifest(self):
        path,sha=self.build();r=p.checked_build(path,sha)
        args=argparse.Namespace(receipt=path,receipt_sha256=sha,output=self.root/'runtime.json',timeout=1)
        from types import SimpleNamespace
        with patch.dict(os.environ,dict(SLURM_JOB_ID='fixture',CUDA_VISIBLE_DEVICES='0')), patch.object(p.os,'uname',return_value=SimpleNamespace(nodename='nid-fixture')), patch.object(p,'checked_build',return_value=r), patch.object(p.subprocess,'run',side_effect=subprocess.TimeoutExpired(['probe'],1)):
            with self.assertRaises(subprocess.TimeoutExpired):p.capture(args)
        self.assertFalse(args.output.exists())
        self.assertTrue((self.root/'runtime.json.capture/failed.json').exists())
        self.assertTrue((self.root/'runtime.json.capture/probe.log').exists())

    def test_native_verifier_rejects_support_library_mutation(self):
        from types import SimpleNamespace
        install=self.root/'install';package=install/'site/psana';package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        (install/'support.so').write_bytes(b'original')
        mpi=self.root/'libmpi.so';mpi.write_bytes(b'fixture MPI')
        runtime=dict(python=sys.executable,python_version=sys.version.split()[0],package=str(package),install=str(install),path_policy='isolated-install-v1',files=p.inventory(package,True),install_files=p.inventory(install),dependencies={},native_libraries={str(mpi):p.file_hash(mpi)},commit=self.sha)
        original_read=Path.read_text
        def maps(path,*a,**kw):
            if str(path)=='/proc/self/maps':return '0-1 r-xp 0 00:00 0 '+str(mpi)+'\n'
            return original_read(path,*a,**kw)
        with patch.dict(sys.modules,{'psana':SimpleNamespace(__file__=str(package/'__init__.py'))}), patch.object(Path,'read_text',maps):
            n.runtime_evidence(runtime)
            with patch.dict(sys.modules,{'psalg':SimpleNamespace(__file__='/other/psalg/__init__.py')}):
                with self.assertRaisesRegex(ValueError,'mixed psalg'):n.runtime_evidence(runtime)
            def wrong_maps(path,*a,**kw):
                if str(path)=='/proc/self/maps':return maps(path)+'0-1 r-xp 0 00:00 0 /other/support.so\n'
                return original_read(path,*a,**kw)
            with patch.object(Path,'read_text',wrong_maps):
                with self.assertRaisesRegex(ValueError,'mixed installed native'):n.runtime_evidence(runtime)
            (install/'support.so').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'support file changed'):n.runtime_evidence(runtime)

    def test_cpu_import_guard_detects_modules_and_native_mappings(self):
        p.check_cpu_imports({'numpy':object()},'')
        for name in ('cupy', 'cupy_backends.cuda', 'kvikio', 'numba.cuda.api', 'cuda.bindings'):
            with self.assertRaisesRegex(ValueError,'GPU module'):p.check_cpu_imports({name:object()},'')
        for name in ('libcuda.so.1','libcudart.so.12','libnvrtc.so','libcufile.so'):
            with self.assertRaisesRegex(ValueError,'CUDA library'):p.check_cpu_imports({},'0-1 r-xp 0 00:00 0 /lib/'+name)

    def test_atomic_publication_never_overwrites_or_publishes_partial_json(self):
        output=self.root/'result.json';p.publish(output,{'ok':True})
        with self.assertRaises(FileExistsError):p.publish(output,{'ok':False})
        self.assertEqual(json.loads(output.read_text()),{'ok':True})
        bad=self.root/'bad.json'
        with self.assertRaises(TypeError):p.publish(bad,{'bad':object()})
        self.assertFalse(bad.exists())

    def test_capture_early_failure_records_reason(self):
        from types import SimpleNamespace
        path,sha=self.build();r=p.checked_build(path,sha)
        args=argparse.Namespace(receipt=path,receipt_sha256=sha,output=self.root/'runtime.json',timeout=1)
        with patch.dict(os.environ,dict(SLURM_JOB_ID='fixture',CUDA_VISIBLE_DEVICES='')), patch.object(p.os,'uname',return_value=SimpleNamespace(nodename='nid-fixture')), patch.object(p,'checked_build',return_value=r):
            with self.assertRaisesRegex(ValueError,'expose the selected GPU'):p.capture(args)
        self.assertTrue((self.root/'runtime.json.capture/failed.json').exists())
        self.assertFalse(args.output.exists())

    def test_capture_publishes_matching_child_and_rejects_association_changes(self):
        from types import SimpleNamespace
        path,sha=self.build();r=p.checked_build(path,sha)
        base=dict(commit=r['commit'],install=r['install'],preparation=dict(build_receipt_sha256=sha))
        for index,change in enumerate((None,'commit','install','receipt')):
            child=copy.deepcopy(base)
            if change=='receipt':child['preparation']['build_receipt_sha256']='bad'
            elif change:child[change]='bad'
            args=argparse.Namespace(receipt=path,receipt_sha256=sha,output=self.root/('runtime%d.json'%index),timeout=1)
            def fake_probe(command,**kwargs):
                Path(command[-1]).write_text(json.dumps(child))
                return subprocess.CompletedProcess(command,0)
            with patch.dict(os.environ,dict(SLURM_JOB_ID='fixture',CUDA_VISIBLE_DEVICES='0')), patch.object(p.os,'uname',return_value=SimpleNamespace(nodename='nid-fixture')), patch.object(p,'checked_build',return_value=r), patch.object(p.subprocess,'run',side_effect=fake_probe):
                if change:
                    with self.assertRaisesRegex(ValueError,'association differs'):p.capture(args)
                    self.assertFalse(args.output.exists())
                else:
                    p.capture(args)
                    self.assertEqual(json.loads(args.output.read_text()),base)

    def test_capture_requires_compute_before_mutation(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'compute allocation'):p.capture(argparse.Namespace())


if __name__=='__main__':unittest.main()

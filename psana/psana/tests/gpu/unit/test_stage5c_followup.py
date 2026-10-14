"""Never submit scaling before the completion and timing reviews succeed."""
import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[3]/'gpu/scripts/stage5c/followup.py'
spec=importlib.util.spec_from_file_location('stage5c_followup',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def setup(monkeypatch,tmp_path,fail=None):
    args=['followup','--root',str(tmp_path),'--campaign',str(tmp_path/'campaign'),'--job','10',
          '--kernel-root',str(tmp_path/'kernel'),'--kernel-job','11',
          '--jf',str(tmp_path/'jf'),'--mixed',str(tmp_path/'mixed')]
    monkeypatch.setattr(module.sys,'argv',args)
    monkeypatch.setattr(module,'verify',lambda root:None)
    monkeypatch.setattr(module,'state',lambda job:dict(job=job,state='FAILED' if fail=='job' else 'COMPLETED',exit_code='0:0'))
    def run(command,stdout,check):
        if fail and command[1].endswith(fail+'.py'):
            raise RuntimeError('review rejected')
        stdout.write(json.dumps(dict(results=[])))
    monkeypatch.setattr(module.subprocess,'run',run)
    submissions=[]
    def submit(command,text):
        submissions.append(command)
        return str(100+len(submissions))+'\n'
    monkeypatch.setattr(module.subprocess,'check_output',submit)
    return submissions


@pytest.mark.parametrize('fail',['job','review','plausibility'])
def test_failure_does_not_submit(monkeypatch,tmp_path,fail):
    submissions=setup(monkeypatch,tmp_path,fail)
    with pytest.raises((AssertionError,RuntimeError)):module.main()
    assert not submissions


def test_success_submits_once_and_rerun_preserves_job_ids(monkeypatch,tmp_path):
    submissions=setup(monkeypatch,tmp_path)
    module.main();module.main()
    assert len(submissions)==2
    launches=json.loads((tmp_path/'launches.json').read_text())
    assert launches['jf']['job']=='101' and launches['mixed']['job']=='102'

"""Freeze reviewed Stage 3 attempts only after data and host checks pass. No submission."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
from compose import compose,HERE
from design import plans,launch,ROOT
from data_contract import verify_inputs,sha,require


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def prepare(qualified):
    require(json.loads((HERE/'status.json').read_text())['submission_authorized'],'Sequential submission authorized')
    info=json.loads((qualified/'inputs.json').read_text());verify_inputs(info)
    receipt=Path(info['qualification']['path'])
    require(sha(receipt)==info['qualification']['sha256'],'Qualification receipt pin')
    require(json.loads(receipt.read_text())['status']=='pass','Qualification pass')
    host=json.loads((HERE/'evidence/host-validation.json').read_text())
    require(host['status']=='pass','Host qualification required')
    repo=Path(subprocess.check_output(['git','-C',str(HERE),'rev-parse','--show-toplevel'],text=True).strip())
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    require(not subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--',str(HERE.relative_to(repo))],text=True),'Uncommitted Stage 3 changes')
    review=json.loads((HERE/'evidence/approved-review.json').read_text())
    require(review['verdict']=='pass' and not any(f['blocking'] for f in review['findings']),'Passing code review required')
    files=compose();hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    require(hashes==host['bundle_hashes'] and hashes==review['bundle_hashes'],'Validated/reviewed bundle unchanged')
    ROOT.mkdir(parents=True,exist_ok=True)
    require(all(not Path(p['attempt']).exists() for p in plans()),'Never overwrite/retry attempts')
    prepared=[]
    for plan in plans():
        attempt=Path(plan['attempt']);attempt.mkdir()
        for name in ('bundle','tmp','cupy-cache'):(attempt/name).mkdir()
        for name,raw in files.items():
            path=attempt/'bundle'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        for name in ('inputs.json','ranges.json'):shutil.copyfile(qualified/name,attempt/name)
        external=[attempt/'inputs.json',attempt/'ranges.json',receipt,Path(info['calibration_selection'])]
        runtime=json.loads(files['runtimes.json'])['C'];external.append(Path(runtime['build_receipt']))
        write(attempt/'bundle/bundle.json',dict(source_commit=commit,files=hashes,external={str(p):sha(p) for p in external}))
        plan.update(source_commit=commit,bundle_sha256=sha(attempt/'bundle/bundle.json'),submission_authorized=True)
        write(attempt/'plan.json',plan);write(attempt/'launch-command.json',launch(plan))
        for p in (attempt/'bundle').rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        (attempt/'bundle').chmod(0o555)
        for n in ('plan.json','launch-command.json','inputs.json','ranges.json'):(attempt/n).chmod(0o444)
        prepared.append(dict(name=plan['name'],attempt=str(attempt),walltime=plan['walltime'],plan_sha256=sha(attempt/'plan.json')))
    write(HERE/'evidence/prepared-attempts.json',prepared)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('qualified_directory');a=parser.parse_args();prepare(Path(a.qualified_directory).resolve())

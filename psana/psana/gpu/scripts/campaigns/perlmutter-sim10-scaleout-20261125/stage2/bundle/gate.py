"""Verify a completed prepared group and enforce prospective predecessor gates."""
from pathlib import Path
import hashlib,json,math,subprocess,sys
ROOT=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-stage2-20261124')
def require(ok,message):
 if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def completed(attempt):
 plan=read(attempt/'plan.json');state=read(attempt/'results/complete.json')
 require(state==read(attempt/'results/status.json'),'Final state changed')
 require(state['status']=='passed_pending_scheduler_review','Complete coordinator')
 require(state['plan_sha256']==sha(attempt/'plan.json'),'Plan pin')
 require([r['case'] for r in state['completed']]==plan['cases'],'Exact ordered case set')
 job=state['job']
 text=subprocess.check_output(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,NodeList'],text=True,timeout=30)
 rows=[line.split('|') for line in text.splitlines() if line.strip()]
 require(any(r[0]==job for r in rows),'Allocation accounting')
 require(all(len(r)>=4 and (r[0]==job or r[0].startswith(job+'.')) and r[1:3]==['COMPLETED','0:0'] and r[3]==state['node'] for r in rows),'Scheduler completion/node')
 for entry in state['completed']:
  name=entry['case']['name'];result=attempt/'results'/(name+'.json')
  require(sha(result)==entry['result_sha256'],'Result pin')
  require(read(attempt/'results'/(name+'.audit.json'))==entry['audit'] and entry['audit']['status']=='pass','Audit pin')
  if entry['case']['numerical']:require(entry['numerical']['status']=='pass','Numerical acceptance')
 summary=read(attempt/'results/summary.json')
 require(sha(attempt/'results/summary.json')==state['summary_sha256'],'Summary pin')
 require(summary['job']==job and summary['node']==state['node'],'Summary identity')
 require(all(abs(d['rate_ratio']-1)<=0.05 for d in summary['drift']),'Staging drift')
 return plan,state,summary

def bundle_identity(plan):
 attempt=Path(plan['attempt']);manifest_path=attempt/'bundle/bundle.json'
 require(sha(manifest_path)==plan['bundle_sha256'],'Bundle manifest pin')
 manifest=read(manifest_path)
 require(manifest['source_commit']==plan['source_commit'],'Bundle source pin')
 return dict(source_commit=plan['source_commit'],files=manifest['files'],inputs=sha(attempt/'inputs.json'),ranges=sha(attempt/'ranges.json'))

def preconditions(plan):
 require(plan['phase']=='paired' and plan['roles']==['P','C'],'Direct pre-sync/candidate screen')
 predecessor=plan['predecessor']
 if predecessor is None:return dict(predecessor=None)
 require(Path(predecessor).name==predecessor,'Predecessor basename')
 previous,state,summary=completed(ROOT/predecessor)
 require(previous['name']==predecessor and previous['phase']=='paired','Predecessor identity')
 require(bundle_identity(previous)==bundle_identity(plan),'Same harness and inputs across all groups')
 return dict(predecessor=predecessor,job=state['job'],status='pass')
if __name__=='__main__':
 print(json.dumps(preconditions(read(Path(sys.argv[1])/'plan.json')),indent=2))

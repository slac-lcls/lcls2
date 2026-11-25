"""Execute one authorized immutable command once; no retries or successor launch."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,socket,subprocess,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def require(ok,message):
 if not ok:raise ValueError(message)
def verify(g):
 a=Path(g['attempt']);require(a.parent==ROOT,'Prepared root')
 for name,key in [('plan.json','plan_sha256'),('launch-command.json','launch_sha256'),('bundle/bundle.json','bundle_sha256')]:require(sha(a/name)==g[key],'Frozen '+name)
 m=read(a/'bundle/bundle.json')
 require(all(sha(a/'bundle'/k)==v for k,v in m['files'].items()),'Bundle files')
 require(all(sha(k)==v for k,v in m['external'].items()),'External inputs')
 return a,read(a/'plan.json')
def main(index):
 authorization=read(HERE/'authorization.json');groups=read(HERE/'prepared-attempts.json')['groups']
 require(0<=index<len(groups)==6,'Group index')
 g=groups[index];a,plan=verify(g);require(plan['source_commit']==authorization['source_commit'],'Authorized source')
 require(not (a/'job-started.txt').exists() and not (a/'results').exists(),'Never relaunch')
 sys.path.insert(0,str(a/'bundle'));from gate import preconditions
 predecessor=preconditions(plan)
 with (ROOT.parent/'sim10-baseline-20261115/count2-use.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 raw=subprocess.check_output(['iris','user',os.environ['USER'],'--json'],text=True,timeout=30)
 save(HERE/('iris-%d.json'%index),json.loads(raw));b=next(r for r in json.loads(raw) if r['Project']=='lcls_g')
 percentages={k:100*(b['Allocated'+k]-b['Charged'+k])/b['Allocated'+k] for k in ('(user)','')}
 require(min(percentages.values())>=20,'Balance below 20%; stop for user alert')
 qos=subprocess.check_output(['sacctmgr','-n','-P','show','qos','interactive','format=Name,MaxWall,MaxTRES'],text=True,timeout=30)
 require(any(r.startswith('interactive|04:00:00|') for r in qos.splitlines()),'Interactive limit changed')
 command=read(a/'launch-command.json')
 record=dict(index=index,name=g['name'],attempt=str(a),command=command,predecessor=predecessor,balance_percent=percentages,qos=qos,hostname=socket.gethostname(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 save(HERE/('launch-%d.json'%index),record)
 started=time.monotonic()
 with (HERE/('salloc-%d.log'%index)).open('x') as log:r=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 record.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 save(HERE/('exit-%d.json'%index),record)
 require(r.returncode==0,'Allocation failed; do not retry')
if __name__=='__main__':main(int(sys.argv[1]))

"""Small host-only metadata preflight. Never hashes payloads or submits work."""
import argparse,datetime,json,os
from pathlib import Path
from data_contract import published_files,validate_site


def inspect(site):
    result=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),site=site,
                status='blocked',files=[],limitations=['No payload hashing, native scan or GPU work on login node'])
    try:
        validate_site(site)
        for rel in published_files():
            p=Path(site['xtc_directory'])/rel;s=p.stat()
            result['files'].append(dict(path=str(p),bytes=s.st_size,uid=s.st_uid,readable=os.access(p,os.R_OK)))
        result['status']='metadata_ready_pending_compute_qualification'
    except (ValueError,OSError,KeyError,TypeError) as error:result['error']=str(error)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('site');parser.add_argument('output');a=parser.parse_args()
    result=inspect(json.loads(Path(a.site).read_text()));Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    raise SystemExit(0 if result['status'].startswith('metadata_ready') else 2)

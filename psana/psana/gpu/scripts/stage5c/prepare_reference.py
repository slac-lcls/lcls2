"""Extend the frozen JF reference using independent SMD records only."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from prepare_feespec_reference import smd_rows
from feespec import coalesced_requests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert a.output.resolve()!=a.reference.resolve()
    refs=json.loads(a.reference.read_text())
    streams={name:smd_rows(a.directory,name,10000)
             for name in refs['10000']['stage_bytes']}
    for n in (200,1000,4000,10000):
        stamps=[r['timestamp'] for r in next(iter(streams.values()))[:n]]
        assert len(stamps)==n
        assert all([r['timestamp'] for r in rows[:n]]==stamps for rows in streams.values())
        result=dict(timestamp_sha256=hashlib.sha256(struct.pack(f'<{n}Q',*stamps)).hexdigest(),
                    payload_bytes=sum(r['size'] for rows in streams.values() for r in rows[:n]))
        if str(n) in refs:
            assert all(refs[str(n)][key]==value for key,value in result.items())
        else:
            if 'feespec' in refs['10000']:
                full=refs['10000']['feespec']
                assert full['timestamps'][:n]==stamps
                result['feespec']=dict(timestamps=stamps,sums=full['sums'][:n],
                    arrays=[x for x in full['arrays'] if x['timestamp'] in set(stamps)])
                result['requests']=dict(off=n*len(streams),
                    on=sum(coalesced_requests(rows[:n]) for rows in streams.values()))
            refs[str(n)]=result
    with a.output.open('x') as f:
        json.dump(refs,f,indent=2);f.write('\n')


if __name__=='__main__':main()

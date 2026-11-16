"""Small controlled two-stream arithmetic test, including malformed-panel rejection."""
import json
from pathlib import Path
import sys
import numpy as np
from perf_cube import Cube
from timing import Recorder


def run():
    import cupy as cp
    results=[]
    for profile in (False,True):
        for bad in (False,True):
            recorder=Recorder(profile)
            cube=Cube(range(8),warmup=2,recorder=recorder,nbins=4,active_bins=3,
                      shape=(2,3,5),max_state_bytes=1<<20)
            cube.prepare()
            streams=[cp.cuda.Stream(non_blocking=True) for _ in range(2)]
            owners=[]
            expected=[np.zeros((4,2,3,5),np.float32) for _ in streams]
            for slot,ts in [(0,[0,3]),(1,[1,4]),(0,[2,5]),(1,[6,7])]:
                values=np.stack([np.full((2,3,5),i+1,np.float32) for i in ts])
                errors=np.zeros((2,2),np.uint32)
                if bad and 5 in ts: errors[ts.index(5),1]=3
                with streams[slot]:
                    images=cp.asarray(values); err=cp.asarray(errors)
                    owners.extend((values,errors,images,err))
                    cube.add(images,err,ts,streams[slot],lambda *a:owners.extend(a))
                for i in ts:
                    if not (bad and i==5):expected[slot][i%3]+=i+1
            cube.join()
            for bank,want in zip(cube.banks,expected):
                assert np.array_equal(bank.sums.get(),want), 'Accumulation changed'
            if bad:
                try: cube.check_counts()
                except RuntimeError: pass
                else: raise AssertionError('Rejected event was not detected')
            else:
                assert cube.check_counts()['counts']==[3,3,2,0]
            if profile:
                assert {r['stage'] for r in recorder.summary()['cuda']}=={'cube_count_kernel','cube_sum_kernel'}
            else:
                assert not recorder.cuda
            cube.close()
            results.append(dict(profile=profile,bad_panel=bad,status='pass'))
    return dict(status='pass',cases=results)


if __name__=='__main__':
    result=run()
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print(result)

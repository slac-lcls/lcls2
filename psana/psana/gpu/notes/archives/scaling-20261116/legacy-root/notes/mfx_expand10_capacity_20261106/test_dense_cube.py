"""Focused configurable budget, streamed export, and stream ownership checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dense_cube import Cube


class HostTests(unittest.TestCase):
    def test_admission(self):
        exact=200*2*(32*512*1024*4+8)
        task=Cube(range(1000),nbins=200,active_bins=199,max_state_bytes=exact)
        self.assertEqual(task.required_state_bytes,exact)
        self.assertEqual(task._identify([999,0,199])[1].tolist(),[4,0,0])
        with self.assertRaises(ValueError):
            Cube([],nbins=200,active_bins=199,max_state_bytes=exact-1)
        for update in ({'nbins':2.5},{'active_bins':201},{'max_streams':3},{'shape':(0,2)}, {'nbins':True}):
            args=dict(nbins=200,active_bins=199,max_state_bytes=exact);args.update(update)
            with self.assertRaises(ValueError):Cube([],**args)


def device_tests(root):
    import cupy as cp
    root=Path(root);root.mkdir(exist_ok=False)
    streams=[cp.cuda.Stream(non_blocking=True) for _ in range(3)]
    owners=[]
    task=Cube(range(9),nbins=40,active_bins=3,shape=(2,3,5),max_state_bytes=1<<20)
    expected=np.zeros((40,2,3,5),np.float64);counts=np.zeros(40,np.uint64)
    for ts,label,slot in [([0,3],10,0),([1,4],10,1),([2,5,6],20,0),([7],20,1)]:
        data=np.stack([np.full((2,3,5),i+1,np.float32) for i in ts]);errors=np.zeros((len(ts),2),np.uint32)
        if 5 in ts:errors[ts.index(5),1]=3
        with streams[slot]:
            device=cp.asarray(data);err=cp.asarray(errors);owners.extend([data,errors,device,err])
            task.add(device,err,ts,streams[slot],lambda *a:owners.extend(a),label)
        for i in ts:
            if i!=5:expected[i%3]+=i+1;counts[i%3]+=1
    with streams[2]:
        data=cp.full((1,2,3,5),9,cp.float32);errors=cp.zeros((1,2),cp.uint32)
    streams[2].synchronize();owners.extend([data,errors])
    before=list(task.banks),set(task.seen)
    try:task.add(data,errors,[8],streams[2],lambda *a:owners.extend(a));raise AssertionError('third stream accepted')
    except RuntimeError:pass
    assert task.banks==before[0] and task.seen==before[1]
    try:task.finish_to(root/'incomplete');raise AssertionError('incomplete accepted')
    except RuntimeError:pass
    task.add(data,errors,[8],streams[0],lambda *a:owners.extend(a),20)
    expected[2]+=9;counts[2]+=1
    assert all(bank.step_labels=={10,20} for bank in task.banks)
    original_save=np.save;ids=[]
    def observed_save(file, array, **kwargs):
        ids.append((id(array),array.shape,array.nbytes));return original_save(file,array,**kwargs)
    with patch('dense_cube.np.save',observed_save):result=task.finish_to(root/'complete')
    assert len(ids)==40 and len({i[0] for i in ids})==1
    assert result['reusable_host_array_bytes']==30*12+40*8
    assert result['counts']==counts.tolist()
    for b in range(40):assert np.array_equal(np.load(root/'complete'/f'bin-{b:05d}.npy'),expected[b])
    assert task.finish_to(root/'complete') is result and task.export_count==1
    task.close();task.close();assert not task.banks
    failure=Cube([],nbins=3,active_bins=2,shape=(2,3,5),max_state_bytes=1<<20)
    save_calls=[]
    def fail_save(*args,**kwargs):
        save_calls.append(1)
        if len(save_calls)==2:raise OSError('injected sink failure')
        return original_save(*args,**kwargs)
    with patch('dense_cube.np.save',fail_save):
        try:failure.finish_to(root/'failed');raise AssertionError('failure hidden')
        except OSError:pass
    assert (root/'failed/bin-00000.npy').is_file()
    assert failure.phase=='failed' and failure.export_count==0 and not (root/'failed/manifest.json').exists()
    try:failure.finish_to(root/'retry');raise AssertionError('retry accepted')
    except RuntimeError:pass
    failure.close()
    fresh=Cube([0],nbins=3,active_bins=2,shape=(2,3,5),max_state_bytes=1<<20)
    fresh.add(data,errors,[0],streams[0],lambda *a:owners.extend(a),99)
    result=fresh.finish_to(root/'fresh');assert result['counts']==[1,0,0]
    assert np.all(np.load(root/'fresh/bin-00000.npy')==9)
    fresh.close()
    aborted=Cube([0,1],nbins=3,active_bins=2,shape=(2,3,5),max_state_bytes=1<<20)
    aborted.add(data,errors,[0],streams[0],lambda *a:owners.extend(a));aborted.close()
    assert not aborted.banks and aborted.export_count==0
    return dict(status='pass',third_stream_rejected_without_mutation=True,repeated_bins_and_invalid_panel=True,
                greater_than_30_bins=True,one_reused_host_bin_buffer=True,all_saved_values_exact=True,
                incomplete_export_rejected=True,idempotent_finish=True,export_failure_no_success_manifest=True,
                fresh_run_isolation=True,abort_join=True)


if __name__=='__main__':unittest.main()

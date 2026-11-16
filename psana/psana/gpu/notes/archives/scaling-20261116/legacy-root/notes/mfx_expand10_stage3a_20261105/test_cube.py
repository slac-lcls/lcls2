"""CPU identity/budget tests and controlled device state/lifetime tests."""
import unittest
import numpy as np
from cube import Cube


class HostTests(unittest.TestCase):
    def test_global_identity(self):
        cube=Cube([10,20,30,40],nbins=3,active_bins=2,shape=(1,2,3))
        ts,bins=cube._identify([40,10,30])
        self.assertEqual(bins.tolist(),[1,0,0])
        cube.seen.update(ts)
        for values in ([10], [50], [20,20], []):
            with self.assertRaises(ValueError): cube._identify(values)

    def test_configuration_budget(self):
        for args in ({'timestamps':[1,1]}, {'timestamps':[], 'active_bins':31},
                     {'timestamps':[1], 'max_state_bytes':10}, {'timestamps':[1], 'max_streams':3}):
            with self.assertRaises(ValueError): Cube(**args)


def device_tests():
    import cupy as cp
    streams=[cp.cuda.Stream(non_blocking=True),cp.cuda.Stream(non_blocking=True)]
    task=Cube(range(8),nbins=4,active_bins=3,shape=(2,3,5))
    owners=[]
    expected=np.zeros((4,2,3,5),np.float64); counts=np.zeros(4,np.uint64)
    for ts,label,slot in [([0,3],10,0),([1,4],10,1),([2,5,6],20,0),([7],20,1)]:
        data=np.stack([np.full((2,3,5),i+1,np.float32) for i in ts])
        errors=np.zeros((len(ts),2),np.uint32)
        if 5 in ts: errors[ts.index(5),1]=3
        with streams[slot]:
            device=cp.asarray(data); err=cp.asarray(errors)
            owners.extend([data,errors,device,err])
            task.add(device,err,ts,streams[slot],lambda *a:owners.extend(a),label)
        for i in ts:
            if i!=5: expected[i%3]+=i+1; counts[i%3]+=1
    bank_info=[{'indices':b.events,'steps':sorted(b.step_labels)} for b in task.banks]
    got,got_counts=task.finish()
    assert np.array_equal(got,expected) and np.array_equal(got_counts,counts)
    assert task.finish()[0] is got and task.export_count==1
    assert all(b.step_labels=={10,20} for b in task.banks)
    try: task._identify([0]); raise AssertionError('accepted process after finish')
    except RuntimeError: pass
    task.close(); task.close()
    assert not task.banks
    # Fresh run, same identities, starts at zero; incomplete runs refuse export.
    fresh=Cube([0],nbins=4,active_bins=3,shape=(2,3,5))
    try: fresh.finish(); raise AssertionError('incomplete export accepted')
    except RuntimeError: pass
    with streams[0]:
        data=cp.ones((1,2,3,5),cp.float32); errors=cp.zeros((1,2),cp.uint32)
        fresh.add(data,errors,[0],streams[0],lambda *a:owners.extend(a),99)
    got,counts=fresh.finish()
    assert np.all(got[0]==1) and not np.any(got[1:]) and counts.tolist()==[1,0,0,0]
    fresh.close()
    aborted=Cube([0,1],nbins=4,active_bins=3,shape=(2,3,5))
    with streams[0]:
        aborted.add(data,errors,[0],streams[0],lambda *a:owners.extend(a),99)
    aborted.close()  # Join submitted work without exporting a partial result.
    assert not aborted.banks and aborted.export_count==0
    empty=Cube([],nbins=4,active_bins=3,shape=(2,3,5))
    a,b=empty.finish(); assert not a.any() and not b.any()
    empty.close()
    return {'status':'pass','step_retention':bank_info,'rejected_missing_panel_event':5,
            'run_isolation':True,'incomplete_export_rejected':True,'abort_join':True,
            'empty_run':True,'idempotent_finish':True,'idempotent_close':True}


if __name__=='__main__': unittest.main()

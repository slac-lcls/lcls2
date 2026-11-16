import unittest
from plan_reads import interleave, run_queue, slab_plan


class ReadPlans(unittest.TestCase):
    def test_slabs_cover_tail_and_round_robin(self):
        rows=slab_plan([10,5],4)
        self.assertEqual(rows,[(0,0,4),(1,0,4),(0,4,4),(1,4,1),(0,8,2)])
        self.assertEqual(sum(r[2] for r in rows),15)

    def test_queue_ownership_and_totals(self):
        busy=set()
        def submit(slot,request):
            self.assertNotIn(slot,busy);busy.add(slot)
            return slot,request
        def complete(future):
            slot,request=future;busy.remove(slot);return request[2]
        rows=slab_plan([10,5],4)
        total,count,last=run_queue(rows,2,submit,complete)
        self.assertEqual((total,count),(15,5));self.assertFalse(busy)
        self.assertEqual(last,{0:rows[4],1:rows[3]})

    def test_short_read_drains_remaining(self):
        completed=[]
        def finish(f):
            completed.append(f)
            return 0 if f==0 else 4
        with self.assertRaisesRegex(ValueError,'Short read'):
            run_queue([(0,i*4,4) for i in range(6)],3,lambda slot,r:r[1]//4,finish)
        self.assertEqual(completed,[0,1,2])

    def test_submit_error_drains_remaining(self):
        completed=[]
        def submit(slot,r):
            if r[1]==8: raise RuntimeError('submit')
            return r[1]
        with self.assertRaisesRegex(RuntimeError,'submit'):
            run_queue([(0,i*4,4) for i in range(4)],3,submit,lambda f:completed.append(f) or 4)
        self.assertEqual(completed,[0,4])

    def test_zero_depth_rejected(self):
        with self.assertRaises(ValueError):run_queue([],0,None,None)


if __name__=='__main__':unittest.main()

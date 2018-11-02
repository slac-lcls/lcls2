import os
import pickle
from psana import DataSource

def test_run_pickle():
    """ Test that run is pickleable for legion """
    xtc_dir = os.path.join(os.getcwd(),'.tmp')
    ds = DataSource('exp=xpptut13:dir=%s'%(xtc_dir), filter=filter)
    run = next(ds.runs())
    run_new = pickle.loads(pickle.dumps(run))
    assert run == run_new

if __name__ == '__main__':
    test_run_pickle()

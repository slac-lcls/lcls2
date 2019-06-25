import os
import pickle
from psana import DataSource

def test_run_pickle(tmp_path=None):
    """ Test that run is pickleable for legion """
    if tmp_path is None:
        xtc_dir = os.path.join(os.getcwd(),'.tmp')
    else:
        xtc_dir = str(tmp_path / '.tmp')
    ds = DataSource(exp='xpptut13', dir=xtc_dir, filter=filter)
    run = next(ds.runs())
    run_new = pickle.loads(pickle.dumps(run))
    assert run == run_new

if __name__ == '__main__':
    test_run_pickle()

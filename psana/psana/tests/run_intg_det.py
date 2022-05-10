from psana import DataSource
import numpy as np
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def intg_det():
    # Set SMD0 and EB batch size
    os.environ['PS_SMD_N_EVENTS'] = '2'
    batch_size = 4

    xtc_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data', 'intg_det')
    ds = DataSource(exp='xpptut15', 
                    run=1, 
                    dir=xtc_dir, 
                    intg_det='andor', 
                    batch_size=batch_size
                   )
    run = next(ds.runs())
    hsd = run.Detector('hsd')
    andor = run.Detector('andor')
    epix = run.Detector('epix')

    # Known values are calculated from the data. Here shown
    # as ts: (andor value, sum(hsd), sum(epix)). 
    known_answers = {7 : (3, 6, 0), 
                     10: (6, 15, 6),
                     13: (9, 24, 0),
                     16: (12, 33, 0),
                     19: (15, 42, 0)
                    }
    sum_hsd = 0
    sum_epix = 0

    for i_evt, evt in enumerate(run.events()):
        hsd_calib = hsd.raw.calib(evt)
        andor_calib = andor.raw.calib(evt)
        epix_calib = epix.raw.calib(evt)
        print(f'RANK:{rank} #evt:{i_evt} ts:{evt.timestamp}')

        # Check that we get correct known values of hsd and epix
        # when andor (integrating detector) events arrive.
        sum_hsd += np.sum(hsd_calib[:])/np.prod(hsd_calib.shape)
        if epix_calib is not None:
            sum_epix += np.sum(epix_calib[:])/np.prod(epix_calib.shape)
        if andor_calib is not None:
            val_andor = np.sum(andor_calib[:])/np.prod(andor_calib.shape)
            print(f'  andor: {val_andor} sum_hsd:{sum_hsd} sum_epix:{sum_epix}')
            if evt.timestamp in known_answers:
                k_andor, k_hsd, k_epix = known_answers[evt.timestamp]
                assert k_andor == val_andor
                assert k_hsd == sum_hsd
                assert k_epix == sum_epix

            sum_hsd = 0
            sum_epix = 0


if __name__ == "__main__":
    intg_det()

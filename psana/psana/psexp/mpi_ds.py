import os
from .ds_base import DataSourceBase
from psana.psexp.run import RunParallel

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class MPIDataSource(DataSourceBase):
    nodetype = "bd"
    nsmds = 1

    def __init__(self, *args, **kwargs):
        expstr = args[0]
        super(MPIDataSource, self).__init__(**kwargs)

        if rank == 0:
            exp, run_dict = super(MPIDataSource, self).parse_expstr(expstr)
        else:
            exp, run_dict = None, None

        assert size >=3
        exp = comm.bcast(exp, root=0)
        run_dict = comm.bcast(run_dict, root=0)

        self.exp = exp
        self.run_dict = run_dict

        # No. of smd cores (default is 1)
        self.nsmds = int(os.environ.get('PS_SMD_NODES', 1))
        if rank == 0:
            self.nodetype = 'smd0'
        elif rank < self.nsmds + 1:
            self.nodetype = 'smd'


    class Factory:
        def create(self, *args, **kwargs): return MPIDataSource(*args, **kwargs)

    def runs(self):
        for run_no in self.run_dict:
            run = RunParallel(self.exp, run_no, self.run_dict[run_no][0], \
                        self.run_dict[run_no][1], self.nodetype, self.nsmds, \
                        self.max_events, self.batch_size, self.filter)
            self.run = run # FIXME: provide support for cctbx code (ds.Detector). will be removed in next cctbx update.
            yield run


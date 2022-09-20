from .datasource import DataSource
#from .smalldata import SmallData


# Collect start-up time (determined as when this file is loaded).
from psana.psexp.prometheus_manager import PrometheusManager
import time
g_ts = PrometheusManager.get_metric('psana_timestamp')
g_ts.labels('psana_init').set(time.time())


# Calls MPI_Abort when one or more (but not all) cores fail.
from psana.psexp.tools import mode
# Checks that we are in MPI and not Legion mode
if mode == 'mpi':
    # We only need the MPI_Abort when working with > 1 core.
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1:
        import sys
        import logging
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(stream=sys.stderr)
        logger.addHandler(handler)

        # Global error handler
        def global_except_hook(exc_type, exc_value, exc_traceback):
            # Needs to write out to logger before calling MPI_Abort
            logger.error("except_hook. Calling MPI_Abort()", exc_info=(exc_type, exc_value, exc_traceback))

            # NOTE: mpi4py must be imported inside exception handler, not globally.
            # In chainermn, mpi4py import is carefully delayed, because
            # mpi4py automatically call MPI_Init() and cause a crash on Infiniband environment.
            import mpi4py.MPI
            mpi4py.MPI.COMM_WORLD.Abort(1)
        sys.excepthook = global_except_hook

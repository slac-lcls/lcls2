from .datasource import DataSource as DataSource

# from .smalldata import SmallData


# Calls MPI_Abort when one or more (but not all) cores fail.
from psana.psexp.tools import mode


xtc_version = 2

# Checks that we are in MPI and not Legion mode
if mode == "mpi":
    # We only need the MPI_Abort when working with > 1 core.
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        import sys
        from psana import utils

        logger = utils.get_logger(name="psana.__init__")

        # Global error handler
        def global_except_hook(exc_type, exc_value, exc_traceback):
            # Needs to write out to logger before calling MPI_Abort
            logger.error(
                "except_hook. Calling MPI_Abort()",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

            # NOTE: mpi4py must be imported inside exception handler, not globally.
            # In chainermn, mpi4py import is carefully delayed, because
            # mpi4py automatically call MPI_Init() and cause a crash on Infiniband environment.
            import mpi4py.MPI

            mpi4py.MPI.COMM_WORLD.Abort(1)

        sys.excepthook = global_except_hook

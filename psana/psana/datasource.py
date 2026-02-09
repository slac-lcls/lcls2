import os
import sys
import gc
import socket
from psana.psexp.tools import mode
from psana import utils
import logging
logger = logging.getLogger(__name__)

world_size = 1
if mode == "mpi":
    from mpi4py import MPI

    world_size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # set a unique jobid (rank 0 process id) for prometheus client
    if rank == 0:
        prometheus_jobid = os.getpid()
    else:
        prometheus_jobid = None
    prometheus_jobid = MPI.COMM_WORLD.bcast(prometheus_jobid, root=0)
    os.environ["PS_PROMETHEUS_JOBID"] = str(prometheus_jobid)
else:
    os.environ["PS_PROMETHEUS_JOBID"] = str(os.getpid())


def _detect_node_count():
    if mode == "mpi":
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            host = socket.gethostname()
            hosts = comm.allgather(host)
            unique_hosts = len(set(hosts))
            if unique_hosts:
                return unique_hosts
        except Exception:
            pass

    env_vars = (
        "PS_NUM_NODES",
        "PS_NODES",
        "SLURM_STEP_NUM_NODES",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NNODES",
    )
    for var in env_vars:
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            count = int(raw)
        except ValueError:
            continue
        if count > 0:
            return count

    return 1


def _ensure_local_eb_nodes():
    env_local = os.environ.get("PS_EB_NODE_LOCAL", "0").strip().lower()
    if env_local not in ("1", "true", "yes", "on"):
        return
    node_count = _detect_node_count()
    if node_count <= 0:
        return
    desired = str(node_count)
    current = os.environ.get("PS_EB_NODES", "1")
    if current == desired:
        return
    os.environ["PS_EB_NODES"] = desired
    should_log = True
    if mode == "mpi":
        try:
            from mpi4py import MPI

            should_log = MPI.COMM_WORLD.Get_rank() == 0
        except Exception:
            should_log = True
    if should_log:
        prev = current if current is not None else "unset"
        logger.info(
            "PS_EB_NODE_LOCAL requires one EB per compute node; overriding PS_EB_NODES %s -> %s",
            prev,
            os.environ.get("PS_EB_NODES", "1"),
        )


def _force_mfx_overrides(exp, kwargs):
    if not exp:
        return
    exp_name = str(exp).lower()
    if not exp_name.startswith("mfx"):
        return
    prev_smd_n_events = os.environ.get("PS_SMD_N_EVENTS")
    prev_eb_nodes = os.environ.get("PS_EB_NODES")
    prev_batch_size = kwargs.get("batch_size")
    batch_size = prev_batch_size if prev_batch_size is not None else 1000
    node_count = _detect_node_count()
    if node_count > 0:
        os.environ["PS_EB_NODES"] = str(node_count)
    os.environ["PS_SMD_N_EVENTS"] = "5000"
    batch_override = batch_size > 10
    if batch_override:
        kwargs["batch_size"] = 1
    should_log = True
    if mode == "mpi":
        try:
            should_log = MPI.COMM_WORLD.Get_rank() == 0
        except Exception:
            should_log = True
    if should_log:
        logger = utils.get_logger(name="DataSource")
        msg = (
            "MFX overrides: PS_EB_NODES=%s (was %s), PS_SMD_N_EVENTS=5000 (was %s)"
            % (
                os.environ.get("PS_EB_NODES", "1"),
                prev_eb_nodes if prev_eb_nodes is not None else "unset",
                prev_smd_n_events if prev_smd_n_events is not None else "unset",
            )
        )
        if batch_override:
            msg += ", batch_size=1 (was %s)" % (prev_batch_size if prev_batch_size is not None else "unset")
        logger.debug(msg)


class InvalidDataSource(Exception):
    pass


from psana.psexp.serial_ds import SerialDataSource
from psana.psexp.singlefile_ds import SingleFileDataSource
from psana.psexp.shmem_ds import ShmemDataSource
from psana.psexp.drp_ds import DrpDataSource
from psana.psexp.null_ds import NullDataSource


def DataSource(*args, **kwargs):
    # force garbage collection to clean up old DataSources, in
    # particular to cause destructors to run to close old shmem msg queues
    gc.collect()
    args = tuple(
        map(str, args)
    )  # Hack: workaround for unicode and str being different types in Python 2

    # ==== shared memory ====
    if "shmem" in kwargs:

        if world_size > 1:

            PS_SRV_NODES = int(os.environ.get("PS_SRV_NODES", "0"))
            if PS_SRV_NODES == world_size:
                raise RuntimeError(
                    "All allocated cores are smalldata servers "
                    "(%d of %d)!" % (PS_SRV_NODES, world_size)
                )

            world_group = MPI.COMM_WORLD.Get_group()
            server_group = world_group.Incl(range(PS_SRV_NODES))
            client_group = world_group.Excl(range(PS_SRV_NODES))
            smalldata_kwargs = {
                "server_group": server_group,
                "client_group": client_group,
            }
            kwargs["smalldata_kwargs"] = smalldata_kwargs

            # create NullDataSource for ranks 0...PS_SRV_NODES-1
            # all others are normal ShmemDataSources
            if rank < PS_SRV_NODES:
                return NullDataSource(*args, **kwargs)
            else:
                return ShmemDataSource(*args, **kwargs)

        else:  # world_size == 1
            return ShmemDataSource(*args, **kwargs)

    # ==== from experiment directory ====
    elif "exp" in kwargs:  # experiment string - assumed multiple files

        _force_mfx_overrides(kwargs.get("exp"), kwargs)

        if mode == "mpi":
            if world_size == 1:
                try:
                  return SerialDataSource(*args, **kwargs)
                except FileNotFoundError as err:
                  logger.error('FileNotFoundError in SerialDataSource for **kwargs: %s\n    %s\n' % (str(kwargs), err))
                  sys.exit(1)
            else:

                # >> these lines are here to AVOID initializing node.comms
                #    that class instance sets up the MPI environment, which
                #    is global... and can interfere with other uses of MPI
                #    (particularly shmem). Therefore we wish to isolate it
                #    as much as possible.
                from psana.psexp.node import Communicators
                from psana.psexp.mpi_ds import MPIDataSource

                _ensure_local_eb_nodes()
                comms = Communicators()

                smalldata_kwargs = {
                    "server_group": comms.srv_group(),
                    "client_group": comms.bd_group(),
                }
                kwargs["smalldata_kwargs"] = smalldata_kwargs

                if comms._nodetype in ["smd0", "eb", "bd"]:
                    return MPIDataSource(comms, *args, **kwargs)
                else:
                    return NullDataSource(*args, **kwargs)

        elif mode == "none":
            return SerialDataSource(*args, **kwargs)

        else:
            raise InvalidDataSource(
                "Incorrect mode. DataSource mode only supports either mpi or none (non parallel mode)."
            )

    # ==== from XTC file(s) ====
    elif "files" in kwargs:  # an xtc file
        return SingleFileDataSource(*args, **kwargs)
    elif "drp" in kwargs:  # the DRP
        return DrpDataSource(*args, **kwargs)
    else:
        raise InvalidDataSource(
            "Expected keyword(s) not found. DataSource requires exp, shmem, or files keywords."
        )

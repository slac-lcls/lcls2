import os

import numpy as np

from psana import utils
from psana.dgram import Dgram
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.events import Events
from psana.psexp.smd_events import SmdEvents
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.tools import mode
from psana.psexp import TransitionId
from psana.psexp.prometheus_manager import get_prom_manager

if mode == "mpi":
    from mpi4py import MPI

import time

DAMAGE_USERBITSHIFT = 12
DAMAGE_VALUEBITMASK = 0x0FFF

# Setting up group communications
# Ex. PS_EB_NODES=3 mpirun -n 13
#       1   4   7   10
#   0   2   5   8   11
#       3   6   9   12
# -smd_group-
#       -bd_main_group-
#       color
#       0   0   0   0
#       1   1   1   1
#       2   2   2   2
#       bd_main_rank        bd_rank
#       0   3   6   9       0   1   2   3
#       1   4   7   10      0   1   2   3
#       2   5   8   11      0   1   2   3


class Communicators(object):
    # Reserved nodes are for external applications (e.g. smalldata
    # servers).  These nodes will do nothing for event/step iterators
    # (see run_node method below).  The "psana_group" consists of
    # all non-reserved ranks and is used for smd0/smd/bd cores.
    comm = None
    world_rank = 0
    world_size = 1
    world_group = None
    psana_comm = None
    psana_rank = -1
    psana_size = 0
    smd_comm = None
    smd_rank = 0
    smd_size = 0
    bd_main_comm = None
    bd_main_rank = 0
    bd_main_size = 0
    bd_rank = 0
    bd_size = 0
    color = 0
    _nodetype = None
    bd_comm = None
    node_comm = None
    node_rank = -1
    node_size = 0
    node_leader_comm = None
    node_leader_rank = -1
    node_leader_size = 0
    _is_node_leader = False

    def __init__(self):
        self.logger = utils.get_logger(name="Communicators")
        self.comm = MPI.COMM_WORLD
        self.world_rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.world_group = self.comm.Get_group()
        self.hostname = MPI.Get_processor_name()

        PS_SRV_NODES = int(os.environ.get("PS_SRV_NODES", 0))
        PS_EB_NODES = int(os.environ.get("PS_EB_NODES", 1))
        self.n_smd_nodes = PS_EB_NODES
        psana_world_size = self.world_size - PS_SRV_NODES

        if (self.world_size - PS_SRV_NODES) < 3:
            raise Exception(
                "Too few MPI cores to run parallel psana."
                "\nYou need 3 + #PS_SRV_NODES (currently: %d)"
                "\n\tCurrent cores:  %d"
                "\n\tRequired:       %d"
                "" % (PS_SRV_NODES, self.world_size, PS_SRV_NODES + 3)
            )

        self.psana_group = self.world_group.Excl(
            range(self.world_size - PS_SRV_NODES, self.world_size)
        )
        self.psana_comm = self.comm.Create(self.psana_group)
        if self.psana_comm != MPI.COMM_NULL:
            self.psana_rank = self.psana_comm.Get_rank()
            self.psana_size = self.psana_comm.Get_size()

        self.node_comm = None
        self.node_leader_comm = None
        self._setup_node_comms()

        self.colocate_non_marching = os.environ.get("PS_EB_NODE_LOCAL", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        self.bd_main_group = self.psana_group.Excl([0])
        self._bd_only_group = None
        self._srv_group = MPI.Group.Difference(self.world_group, self.psana_group)

        self.smd_group = None
        self.smd_comm = MPI.COMM_NULL
        self.bd_main_comm = self.comm.Create(self.bd_main_group)

        if self.bd_main_comm != MPI.COMM_NULL:
            self.bd_main_rank = self.bd_main_comm.Get_rank()
            self.bd_main_size = self.bd_main_comm.Get_size()

            if self.colocate_non_marching:
                info = MPI.INFO_NULL
                node_comm = self.bd_main_comm.Split_type(
                    MPI.COMM_TYPE_SHARED, self.bd_main_rank, info
                )
                if self.bd_main_rank == 0:
                    self.logger.info(
                        f'[MPI-role] Non-marching EB/BD colocated mode enabled on host {self.hostname}'
                    )
                self.bd_comm = node_comm
            else:
                color = self.bd_main_rank % PS_EB_NODES
                self.bd_comm = self.bd_main_comm.Split(color, self.bd_main_rank)

            if self.bd_comm == MPI.COMM_NULL:
                raise RuntimeError("Failed to create bd_comm")
            self.bd_rank = self.bd_comm.Get_rank()
            self.bd_size = self.bd_comm.Get_size()

            if self.bd_rank == 0:
                self._nodetype = "eb"
            else:
                self._nodetype = "bd"

        if self.world_rank == 0:
            self._nodetype = "smd0"
        elif self.world_rank >= self.psana_group.Get_size():
            self._nodetype = "srv"

        # Ensure every rank has finalized its node role before building
        # the SMD communicator, since _init_smd_comm gathers the EB list.
        self.comm.Barrier()
        self._init_smd_comm(PS_EB_NODES, psana_world_size)
        self._report_role()
        self._report_smd_members()

    def _setup_node_comms(self):
        """Split psana ranks into per-node and node-leader communicators."""
        if self.psana_comm == MPI.COMM_NULL:
            return

        hostname = MPI.Get_processor_name()
        hostnames = self.psana_comm.allgather(hostname)
        color_map = {}
        colors = []
        for idx, host in enumerate(hostnames):
            if host not in color_map:
                color_map[host] = len(color_map)
            colors.append(color_map[host])

        node_color = colors[self.psana_rank]
        self.node_comm = self.psana_comm.Split(node_color, self.psana_rank)
        self.node_rank = self.node_comm.Get_rank()
        self.node_size = self.node_comm.Get_size()
        self._is_node_leader = self.node_rank == 0

        leader_color = 0 if self._is_node_leader else MPI.UNDEFINED
        self.node_leader_comm = self.psana_comm.Split(leader_color, self.psana_rank)
        if self.node_leader_comm != MPI.COMM_NULL:
            self.node_leader_rank = self.node_leader_comm.Get_rank()
            self.node_leader_size = self.node_leader_comm.Get_size()

    def bd_group(self):
        return self._bd_only_group

    def srv_group(self):
        return self._srv_group

    def node_type(self):
        return self._nodetype

    def is_node_leader(self):
        return self._is_node_leader

    def get_node_comm(self):
        return self.node_comm

    def get_node_leader_comm(self):
        return self.node_leader_comm

    def terminate(self):
        """Tells Smd0 to terminate the loop.

        Smd0 is waiting (non-blocking) on the world comm for a signal (just
        world rank no. (so we know who requests the termination).
        """
        self.comm.Isend(np.array([self.world_rank], dtype="i"), dest=0)

    def _report_role(self):
        role = self._nodetype or "unknown"
        role_upper = role.upper()
        if role == "eb":
            detail = f"bd_main_rank={getattr(self, 'bd_main_rank', -1)}"
        elif role == "bd":
            detail = f"bd_rank={getattr(self, 'bd_rank', -1)}"
        else:
            detail = ""
        if detail:
            message = f"[MPI-role] rank={self.world_rank} host={self.hostname} role={role_upper} {detail}"
        else:
            message = f"[MPI-role] rank={self.world_rank} host={self.hostname} role={role_upper}"
        self.logger.debug(message)

    def _report_smd_members(self):
        if self.smd_comm is None or self.smd_comm == MPI.COMM_NULL:
            return
        members = []
        if self.smd_comm != MPI.COMM_NULL:
            members = self.smd_comm.allgather((self.world_rank, self.hostname))
        total = self.smd_comm.Get_size() if self.smd_comm != MPI.COMM_NULL else 0
        if self.world_rank == 0:
            pretty = ", ".join(f"{rank}@{host}" for rank, host in members)
            self.logger.debug(
                "[MPI-role] smd_comm size=%d members=[%s]", total, pretty
            )

    def _init_smd_comm(self, ps_eb_nodes, psana_world_size):
        if ps_eb_nodes < 1:
            ps_eb_nodes = 1
        if self.colocate_non_marching:
            eb_candidate = self.world_rank if self._nodetype == "eb" else -1
            gathered = self.comm.allgather(eb_candidate)
            if self.world_rank == 0:
                eb_worlds = sorted({r for r in gathered if r >= 0})
                if not eb_worlds:
                    raise RuntimeError(
                        "PS_EB_NODE_LOCAL requires at least one EB node but none were detected"
                    )
                smd_worlds = [0] + eb_worlds
            else:
                smd_worlds = None
            smd_worlds = self.comm.bcast(smd_worlds, root=0)
        else:
            count = min(ps_eb_nodes + 1, self.psana_group.Get_size())
            smd_worlds = list(range(count))

        translated = MPI.Group.Translate_ranks(
            self.world_group, smd_worlds, self.psana_group
        )
        smd_indices = [idx for idx in translated if idx != MPI.UNDEFINED]
        if not smd_indices:
            smd_indices = [0]

        self.smd_group = self.psana_group.Incl(smd_indices)

        color = 0 if self.world_rank in smd_worlds else MPI.UNDEFINED
        self.smd_comm = self.comm.Split(color, self.world_rank)

        if self.smd_comm != MPI.COMM_NULL:
            self.smd_rank = self.smd_comm.Get_rank()
            self.smd_size = self.smd_comm.Get_size()
        self._bd_only_group = MPI.Group.Difference(self.bd_main_group, self.smd_group)


class StepHistory(object):
    """Keeps step data and their send history."""

    def __init__(self, client_size, n_smds):
        self.n_smds = n_smds
        self.bufs = []
        for i in range(self.n_smds):
            self.bufs.append(bytearray())
        self.send_history = []
        # Initialize no. of sent bytes to 0 for clients
        # [[offset_update0, offset_update1, ], [offset_update0, offset_update1, ], ...]
        # [ -----------client 0------------- ,  ----------- client 1------------ ,
        for i in range(1, client_size):
            self.send_history.append(np.zeros(self.n_smds, dtype=int))

    def extend_buffers(self, views, client_id, as_event=False):
        idx = client_id - 1  # rank 0 has no send history.
        # Views is either list of smdchunks or events
        if not as_event:
            # For Smd0
            for i_smd, view in enumerate(views):
                self.bufs[i_smd].extend(view)
                self.send_history[idx][i_smd] += view.nbytes
        else:
            # For EventBuilder
            for i_evt, evt_bytes in enumerate(views):
                pf = PacketFooter(view=evt_bytes)
                assert pf.n_packets == self.n_smds
                for i_smd, dg_bytes in enumerate(pf.split_packets()):
                    self.bufs[i_smd].extend(dg_bytes)
                    self.send_history[idx][i_smd] += dg_bytes.nbytes

    def update_history(self, views, client_id):
        indexed_id = client_id - 1  # rank 0 has no send history.
        for i, view in enumerate(views):
            self.send_history[indexed_id][i] += view.nbytes

    def get_buffer(self, client_id, smd0=False):
        """Returns new step data (if any) for this client
        then updates the sent record."""
        views = []

        if self.n_smds:  # do nothing if no step data found
            indexed_id = client_id - 1  # rank 0 has no send history.
            views = [bytearray() for i in range(self.n_smds)]
            for i, buf in enumerate(self.bufs):
                current_buf = self.bufs[i]
                current_offset = self.send_history[indexed_id][i]
                current_buf_size = memoryview(current_buf).nbytes
                if current_offset < current_buf_size:
                    views[i].extend(current_buf[current_offset:])
                    self.send_history[indexed_id][i] = current_buf_size
        return views


def repack_for_bd(smd_batch, step_views, configs, client=-1):
    """EventBuilder Node uses this to prepend missing step views
    to the smd_batch. This output chunk contains list of pre-built events."""
    if step_views:
        batch_pf = PacketFooter(view=smd_batch)

        # Create bytearray containing a list of events from step_views
        steps = bytearray()
        n_smds = len(step_views)
        offsets = np.zeros(n_smds, dtype=int)
        n_steps = 0
        step_sizes = []
        while offsets[0] < memoryview(step_views[0]).nbytes:
            step_pf = PacketFooter(n_packets=n_smds)
            step_size = 0
            for i, (config, view) in enumerate(zip(configs, step_views)):
                d = Dgram(config=config, view=view, offset=offsets[i])
                steps.extend(d)
                offsets[i] += d._size
                step_size += d._size
                step_pf.set_size(i, d._size)

            steps.extend(step_pf.footer)
            step_sizes.append(step_size + memoryview(step_pf.footer).nbytes)
            n_steps += 1

        # Create new batch with total_events = smd_batch_events + step_events
        new_batch_pf = PacketFooter(n_packets=batch_pf.n_packets + n_steps)
        for i in range(n_steps):
            new_batch_pf.set_size(i, step_sizes[i])

        for i in range(n_steps, new_batch_pf.n_packets):
            new_batch_pf.set_size(i, batch_pf.get_size(i - n_steps))

        new_batch = bytearray()
        new_batch.extend(steps)
        new_batch.extend(
            smd_batch[
                : memoryview(smd_batch).nbytes - memoryview(batch_pf.footer).nbytes
            ]
        )
        new_batch.extend(new_batch_pf.footer)
        return new_batch
    else:
        return smd_batch


def wait_for(requests):
    status = [MPI.Status() for i in range(len(requests))]
    MPI.Request.Waitall(requests, status)


class Smd0(object):
    """Sends blocks of smds to eb nodes
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp.
    """

    def __init__(self, comms, smdr_man, configs):
        self.comms = comms
        self.smdr_man = smdr_man
        self.configs = configs
        self.step_hist = StepHistory(self.comms.smd_size, len(self.configs))

        # Collecting Smd0 performance using prometheus
        pm = get_prom_manager()
        self.wait_gauge = pm.get_metric("psana_smd0_wait")
        self.rate_gauge = pm.get_metric("psana_smd0_rate")

        self.logger = utils.get_logger(name=utils.get_class_name(self))
        self._eb_wait_s = []

    def _request_rank(self, rankreq):
        st_req = time.monotonic()
        self.logger.debug(f"TIMELINE 1. SMD0REQ4EB {st_req}")

        req = self.comms.smd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
        req.Wait()
        en_req = time.monotonic()
        self.logger.debug(f"TIMELINE 2. SMD0GOTEB{rankreq[0]} {en_req}")
        self._eb_wait_s.append(en_req - st_req)
        self.wait_gauge.set(en_req - st_req)

    def _reset_smd0_chunk_stats(self):
        self._eb_wait_s = []

    def _log_smd0_chunk_stats(self, chunk_id, read_stats, chunk_rate_khz):
        bytes_list, times_list = read_stats
        wait = np.asarray(self._eb_wait_s, dtype=np.float64)
        read_bytes = np.asarray(bytes_list, dtype=np.float64)
        read_times = np.asarray(times_list, dtype=np.float64)
        read_mb = read_bytes / 1e6 if read_bytes.size else np.asarray([], dtype=np.float64)
        rate_mask = read_times > 0
        read_rates = read_mb[rate_mask] / read_times[rate_mask] if rate_mask.any() else None
        total_read_mb = float(read_mb.sum()) if read_mb.size else 0.0
        total_read_time = float(read_times.sum()) if read_times.size else 0.0
        total_read_rate = (
            total_read_mb / total_read_time if total_read_time > 0 else 0.0
        )
        wait_avg = float(wait.mean()) if wait.size else 0.0
        wait_min = float(wait.min()) if wait.size else 0.0
        wait_max = float(wait.max()) if wait.size else 0.0
        if read_rates is not None:
            rate_avg = float(read_rates.mean())
            rate_min = float(read_rates.min())
            rate_max = float(read_rates.max())
        else:
            rate_avg = rate_min = rate_max = 0.0
        if total_read_mb <= 0:
            return
        self.logger.debug(
            "SMD0 chunk stats chunk=%d\n"
            "  eb_wait_s avg=%.5f min=%.5f max=%.5f count=%d\n"
            "  smd_read_mb avg=%.2f min=%.2f max=%.2f total=%.2f\n"
            "  smd_read_rate_mb_s avg=%.2f min=%.2f max=%.2f total=%.2f\n"
            "  smd0_rate_khz=%.5f",
            chunk_id,
            wait_avg,
            wait_min,
            wait_max,
            wait.size,
            float(read_mb.mean()) if read_mb.size else 0.0,
            float(read_mb.min()) if read_mb.size else 0.0,
            float(read_mb.max()) if read_mb.size else 0.0,
            total_read_mb,
            rate_avg,
            rate_min,
            rate_max,
            total_read_rate,
            chunk_rate_khz,
        )

    def start(self):
        # Rank 0 waits on World comm for terminating signal
        t_rankreq = np.empty(1, dtype="i")
        t_req = self.comms.comm.Irecv(t_rankreq, source=MPI.ANY_SOURCE)

        # Setup a non-pickled recv array and prepare bucket for storing send reqs.
        rankreq = np.empty(1, dtype="i")
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]

        # Need this for async MPI to prevent overwriting send buffer
        repack_smds = {}

        eb_request_queue = []

        def _await_request():
            self._request_rank(rankreq)
            eb_request_queue.append(int(rankreq[0]))

        def _drain_requests():
            smd_comm = self.comms.smd_comm
            while smd_comm.Iprobe(source=MPI.ANY_SOURCE):
                self._request_rank(rankreq)
                eb_request_queue.append(int(rankreq[0]))

        for _ in range(max(1, self.comms.n_smd_nodes)):
            _await_request()

        # Indentify viewing windows. SmdReaderManager has starting index and block size
        # that it needs to share later when data are packaged for sending to EventBuilders.
        for i_chunk in self.smdr_man.chunks():
            self._reset_smd0_chunk_stats()
            st = time.monotonic()
            _drain_requests()
            while not eb_request_queue:
                _await_request()
                _drain_requests()
            rankreq[0] = eb_request_queue.pop(0)

            # Check missing steps for the current client
            missing_step_views = self.step_hist.get_buffer(rankreq[0], smd0=True)
            self.logger.debug(f"TIMELINE 2.1 SMD0GOTSTEPHIST {time.monotonic()}")

            # Update step buffers (after getting the missing steps
            step_views = [
                self.smdr_man.smdr.show(i, step_buf=True)
                for i in range(self.smdr_man.n_files)
            ]
            self.step_hist.extend_buffers(step_views, rankreq[0])
            self.logger.debug(f"TIMELINE 2.2 SMD0STEPHISTUPDATED {time.monotonic()}")

            # Prevent race condition by making a copy of data
            repack_smds[rankreq[0]] = self.smdr_man.smdr.repack_parallel(
                missing_step_views,
                rankreq[0],
                intg_stream_id=self.smdr_man.dsparms.intg_stream_id,
            )

            self.logger.debug(f"TIMELINE 3. SMD0GOTREPACK {time.monotonic()}")

            requests[rankreq[0] - 1] = self.comms.smd_comm.Isend(
                repack_smds[rankreq[0]], dest=rankreq[0]
            )

            # Queue up the next request from an EB to keep assignments rotating
            _drain_requests()
            if not eb_request_queue:
                _await_request()

            self.logger.debug(f"TIMELINE 4. SMD0DONEWITHEB{rankreq[0]} {time.monotonic()}")

            en = time.monotonic()
            chunk_rate_khz = (self.smdr_man.got_events / (en - st)) * 1e-3
            self.rate_gauge.set(chunk_rate_khz)
            read_stats = self.smdr_man.pop_read_stats()
            self._log_smd0_chunk_stats(i_chunk, read_stats, chunk_rate_khz)

            # Check for terminating signal
            t_req_test = t_req.Test()
            if t_req_test:
                self.logger.debug(
                    f"MESSAGE SMD0-TERM-FROMRANK{t_rankreq[0]} (t_req_test:{t_req_test})"
                )
                break

            found_endrun = self.smdr_man.smdr.found_endrun()
            if found_endrun:
                self.logger.debug("MESSAGE SMD0-ENDRUN")
                break

        _drain_requests()

        # end for (smd_chunk, step_chunk)
        wait_for(requests)

        # check if there are missing steps to be sent
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]
        remaining = self.comms.n_smd_nodes
        while remaining > 0:
            _drain_requests()
            if eb_request_queue:
                rankreq[0] = eb_request_queue.pop(0)
            else:
                self._request_rank(rankreq)
            missing_step_views = self.step_hist.get_buffer(rankreq[0], smd0=True)
            repack_smds[rankreq[0]] = self.smdr_man.smdr.repack_parallel(
                missing_step_views,
                rankreq[0],
                only_steps=1,
                intg_stream_id=self.smdr_man.dsparms.intg_stream_id,
            )
            if memoryview(repack_smds[rankreq[0]]).nbytes > 0:
                requests[rankreq[0] - 1] = self.comms.smd_comm.Isend(
                    repack_smds[rankreq[0]], dest=rankreq[0]
                )
            remaining -= 1
        wait_for(requests)

        # kill waiting eb nodes
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]
        for dest_rank in range(1, self.comms.smd_size):
            requests[dest_rank - 1] = self.comms.smd_comm.Isend(
                bytearray(), dest=dest_rank
            )
        wait_for(requests)


class EventBuilderNode(object):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""

    def __init__(self, comms, configs, dsparms):
        self.comms = comms
        self.configs = configs
        self.dsparms = dsparms
        self.step_hist = StepHistory(self.comms.bd_size, len(self.configs))
        pm = get_prom_manager()
        self.rate_gauge = pm.get_metric("psana_eb_rate")
        self.wait_smd0_gauge = pm.get_metric("psana_eb_wait_smd0")
        self.wait_bd_gauge = pm.get_metric("psana_eb_wait_bd")
        self.bd_read_gauge = pm.get_metric("psana_bd_read")
        self.bd_rate_gauge = pm.get_metric("psana_bd_rate")
        self.requests = []
        self.logger = utils.get_logger(name=utils.get_class_name(self))
        self._bd_chunk_stats = None
        self._bd_pending_chunk_id = None
        self._smd_wait_s = []
        self._smd_chunk_bytes = []
        self._bd_wait_s = []
        self._bd_rate_hz = []

    def _init_requests(self):
        self.requests = [MPI.REQUEST_NULL for i in range(self.comms.bd_size - 1)]

    def pack(self, *args):
        pf = PacketFooter(len(args))
        batch = bytearray()
        for i, arg in enumerate(args):
            pf.set_size(i, memoryview(arg).shape[0])
            batch += arg
        batch += pf.footer
        return batch

    def _send_to_dest(
        self, dest_rank, smd_batch_dict, step_batch_dict, eb_man, batches
    ):
        bd_comm = self.comms.bd_comm
        smd_batch, _ = smd_batch_dict[dest_rank]
        missing_step_views = self.step_hist.get_buffer(dest_rank)
        batches[dest_rank] = repack_for_bd(
            smd_batch, missing_step_views, self.configs, client=dest_rank
        )
        self.requests[dest_rank - 1] = bd_comm.Isend(batches[dest_rank], dest=dest_rank)
        del smd_batch_dict[dest_rank]  # done sending

        step_batch, _ = step_batch_dict[dest_rank]
        if eb_man.eb.nsteps > 0 and memoryview(step_batch).nbytes > 0:
            step_pf = PacketFooter(view=step_batch)
            self.step_hist.extend_buffers(
                step_pf.split_packets(), dest_rank, as_event=True
            )
        del step_batch_dict[dest_rank]  # done adding

    def _request_rank(self, rankreq):
        st_req = time.monotonic()
        req = self.comms.bd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
        req.Wait()
        en_req = time.monotonic()
        self._update_bd_read_stats(rankreq)
        self._bd_wait_s.append(en_req - st_req)
        self.wait_bd_gauge.set(en_req - st_req)

    def _update_bd_read_stats(self, rankreq):
        if self._bd_chunk_stats is None or self._bd_pending_chunk_id is None:
            return
        if self._bd_pending_chunk_id < 0:
            return
        bd_rank = int(rankreq[0])
        if bd_rank <= 0:
            return
        idx = bd_rank - 1
        if idx >= self._bd_chunk_stats["bytes"].size:
            return
        self._bd_chunk_stats["bytes"][idx] += int(rankreq[1])
        self._bd_chunk_stats["time_ns"][idx] += int(rankreq[2])
        self._bd_chunk_stats["wait_ns"][idx] += int(rankreq[3])
        self._bd_chunk_stats["proc_events"][idx] += int(rankreq[4])
        self._bd_chunk_stats["proc_time_ns"][idx] += int(rankreq[5])

    def _prepare_bd_read_stats(self, pending_chunk_id, n_bd_nodes):
        if pending_chunk_id < 0 or n_bd_nodes <= 0:
            self._bd_pending_chunk_id = pending_chunk_id
            self._bd_chunk_stats = None
            return
        self._bd_pending_chunk_id = pending_chunk_id
        self._bd_chunk_stats = {
            "bytes": np.zeros(n_bd_nodes, dtype=np.int64),
            "time_ns": np.zeros(n_bd_nodes, dtype=np.int64),
            "wait_ns": np.zeros(n_bd_nodes, dtype=np.int64),
            "proc_events": np.zeros(n_bd_nodes, dtype=np.int64),
            "proc_time_ns": np.zeros(n_bd_nodes, dtype=np.int64),
        }

    def _format_bd_read_stats(self, chunk_id):
        if self._bd_chunk_stats is None or chunk_id < 0:
            return None
        bytes_arr = self._bd_chunk_stats["bytes"]
        time_arr = self._bd_chunk_stats["time_ns"]
        wait_arr = self._bd_chunk_stats["wait_ns"]
        proc_events_arr = self._bd_chunk_stats["proc_events"]
        proc_time_arr = self._bd_chunk_stats["proc_time_ns"]
        if bytes_arr.size == 0:
            return None
        zero_mask = bytes_arr == 0
        zero_chunks = int(zero_mask.sum())
        if zero_mask.all():
            return
        bytes_mb = bytes_arr[~zero_mask] / 1e6
        time_s = time_arr[~zero_mask] / 1e9
        wait_s = wait_arr[~zero_mask] / 1e9
        proc_events = proc_events_arr[~zero_mask]
        proc_time_s = proc_time_arr[~zero_mask] / 1e9
        total_bytes = float(bytes_arr[~zero_mask].sum())
        max_time = float(time_s.max()) if time_s.size else 0.0
        rate_mask = time_s > 0
        rate_mb_s = bytes_mb[rate_mask] / time_s[rate_mask] if rate_mask.any() else None
        total_rate = (total_bytes / 1e6) / max_time if max_time > 0 else 0.0
        if rate_mb_s is not None:
            rate_avg = float(rate_mb_s.mean())
            rate_min = float(rate_mb_s.min())
            rate_max = float(rate_mb_s.max())
            if total_rate > 0:
                self.bd_read_gauge.set(total_rate)
        else:
            rate_avg = rate_min = rate_max = 0.0
        proc_mask = proc_time_s > 0
        proc_total = 0.0
        if proc_mask.any():
            proc_rate = proc_events[proc_mask] / proc_time_s[proc_mask]
            proc_avg = float(proc_rate.mean())
            proc_min = float(proc_rate.min())
            proc_max = float(proc_rate.max())
            proc_total = float(proc_rate.sum())
            if proc_total > 0:
                self.bd_rate_gauge.set(proc_total)
        else:
            proc_avg = proc_min = proc_max = 0.0
        return (
            "EB bd read stats chunk(prev)=%d bds=%d zero_chunks=%d\n"
            "  bytes_per_pread_mb avg=%.2f min=%.2f max=%.2f\n"
            "  wait_s    avg=%.3f min=%.3f max=%.3f\n"
            "  proc_rate_hz avg=%.2f min=%.2f max=%.2f total=%.2f\n"
            "  rate_mb_s avg=%.2f min=%.2f max=%.2f\n"
            "  total_rate=%.2f MB/s"
            % (
                chunk_id,
                bytes_arr.size,
                zero_chunks,
                float(bytes_mb.mean()),
                float(bytes_mb.min()),
                float(bytes_mb.max()),
                float(wait_s.mean()),
                float(wait_s.min()),
                float(wait_s.max()),
                proc_avg,
                proc_min,
                proc_max,
                proc_total,
                rate_avg,
                rate_min,
                rate_max,
                total_rate,
            )
        )

    def _reset_eb_chunk_stats(self):
        self._smd_wait_s = []
        self._smd_chunk_bytes = []
        self._bd_wait_s = []
        self._bd_rate_hz = []

    def _format_eb_chunk_stats(self, chunk_id):
        if chunk_id < 0:
            return None
        wait = np.asarray(self._smd_wait_s, dtype=np.float64)
        sizes = np.asarray(self._smd_chunk_bytes, dtype=np.float64) / 1e6
        bd_wait = np.asarray(self._bd_wait_s, dtype=np.float64)
        bd_rate = np.asarray(self._bd_rate_hz, dtype=np.float64)
        if not (wait.size or bd_wait.size or bd_rate.size):
            return None
        wait_avg = float(wait.mean()) if wait.size else 0.0
        wait_min = float(wait.min()) if wait.size else 0.0
        wait_max = float(wait.max()) if wait.size else 0.0
        size_avg = float(sizes.mean()) if sizes.size else 0.0
        size_min = float(sizes.min()) if sizes.size else 0.0
        size_max = float(sizes.max()) if sizes.size else 0.0
        bd_wait_avg = float(bd_wait.mean()) if bd_wait.size else 0.0
        bd_wait_min = float(bd_wait.min()) if bd_wait.size else 0.0
        bd_wait_max = float(bd_wait.max()) if bd_wait.size else 0.0
        bd_rate_avg = float(bd_rate.mean()) if bd_rate.size else 0.0
        bd_rate_min = float(bd_rate.min()) if bd_rate.size else 0.0
        bd_rate_max = float(bd_rate.max()) if bd_rate.size else 0.0
        bd_rate_med = float(np.median(bd_rate)) if bd_rate.size else 0.0
        return (
            "EB chunk stats chunk=%d\n"
            "  smd_wait_s avg=%.5f min=%.5f max=%.5f count=%d\n"
            "  smd_size_mb avg=%.2f min=%.2f max=%.2f\n"
            "  bd_wait_s  avg=%.5f min=%.5f max=%.5f count=%d\n"
            "  bd_rate_hz avg=%.2f med=%.2f min=%.2f max=%.2f count=%d"
            % (
                chunk_id,
                wait_avg,
                wait_min,
                wait_max,
                wait.size,
                size_avg,
                size_min,
                size_max,
                bd_wait_avg,
                bd_wait_min,
                bd_wait_max,
                bd_wait.size,
                bd_rate_avg,
                bd_rate_med,
                bd_rate_min,
                bd_rate_max,
                bd_rate.size,
            )
        )

    def _log_chunk_stats(self, chunk_id):
        parts = []
        eb_stats = self._format_eb_chunk_stats(chunk_id)
        if eb_stats:
            parts.append(eb_stats)
        bd_stats = self._format_bd_read_stats(self._bd_pending_chunk_id)
        if bd_stats:
            parts.append(bd_stats)
        if bd_stats:
            self.logger.debug("\n".join(parts))

    def _request_data(self, smd_comm):
        st = time.monotonic()
        self.logger.debug(
            f"TIMELINE 5. EB{self.comms.world_rank}SENDREQTOSMD0 {time.monotonic()}",
        )
        smd_comm.Isend(np.array([self.comms.smd_rank], dtype="i"), dest=0)
        self.logger.debug(
            f"TIMELINE 6. EB{self.comms.world_rank}DONESENDREQ {time.monotonic()}",
        )
        info = MPI.Status()
        smd_comm.Probe(source=0, status=info)
        count = info.Get_elements(MPI.BYTE)
        smd_chunk = bytearray(count)
        req = smd_comm.Irecv(smd_chunk, source=0)
        req.Wait()
        en = time.monotonic()
        self.logger.debug(
            f"TIMELINE 7. EB{self.comms.world_rank}RECVDATA {time.monotonic()}"
        )
        self._smd_wait_s.append(en - st)
        self._smd_chunk_bytes.append(count)
        self.wait_smd0_gauge.set(en - st)
        return smd_chunk

    def start(self):
        rankreq = np.empty(6, dtype=np.int64)
        smd_comm = self.comms.smd_comm
        n_bd_nodes = self.comms.bd_comm.Get_size() - 1
        bd_comm = self.comms.bd_comm
        waiting_bds = []
        chunk_id = 0

        # Initialize Non-blocking Send Requests with Null
        self._init_requests()

        bypass_bd = bool(int(os.environ.get("PS_EB_BYPASS_BD", "0")))

        while True:
            self._reset_eb_chunk_stats()
            smd_chunk = self._request_data(smd_comm)
            if not smd_chunk:
                break

            if not bypass_bd:
                self._prepare_bd_read_stats(chunk_id - 1, n_bd_nodes)
            eb_man = EventBuilderManager(
                smd_chunk, self.configs, self.dsparms,
            )
            self.logger.debug(
                f"TIMELINE 8. EB{self.comms.world_rank}DONEBUILDINGEVENTS {time.monotonic()}",
            )

            # Build batches of events

            # Need this for async MP to prevent overwriting send buffer
            # The key of batches dict is the bd rank.
            batches = {}

            t0 = time.monotonic()
            for smd_batch_dict, step_batch_dict in eb_man.batches():
                if bypass_bd:
                    continue
                # If single item and dest_rank=0, send to any bigdata nodes.
                if 0 in smd_batch_dict.keys():
                    smd_batch, _ = smd_batch_dict[0]
                    step_batch, _ = step_batch_dict[0]

                    self.logger.debug(
                        f"TIMELINE 9. EB{self.comms.world_rank}REQBD {time.monotonic()}",
                    )
                    if waiting_bds:
                        rankreq[0] = waiting_bds.pop()
                        self.logger.debug(
                            f"TIMELINE 10. EB{self.comms.world_rank}GOTBD{rankreq[0]+1}FROMQUEUE {time.monotonic()}",
                        )
                    else:
                        self._request_rank(rankreq)
                        self.logger.debug(
                            f"TIMELINE 10. EB{self.comms.world_rank}GOTBD{rankreq[0]+1}FROMREQ {time.monotonic()}",
                        )

                    missing_step_views = self.step_hist.get_buffer(rankreq[0])
                    batches[rankreq[0]] = repack_for_bd(
                        smd_batch, missing_step_views, self.configs, client=rankreq[0]
                    )

                    self.logger.debug(
                        f"TIMELINE 11. EB{self.comms.world_rank}SENDDATATOBD{rankreq[0]+1} {time.monotonic()}",
                    )
                    self.requests[rankreq[0] - 1] = bd_comm.Isend(
                        batches[rankreq[0]], dest=rankreq[0]
                    )
                    self.logger.debug(
                        f"TIMELINE 12. EB{self.comms.world_rank}DONESENDDATATOBD{rankreq[0]+1} {time.monotonic()}",
                    )

                    if eb_man.eb.nsteps > 0 and memoryview(step_batch).nbytes > 0:
                        step_pf = PacketFooter(view=step_batch)
                        self.step_hist.extend_buffers(
                            step_pf.split_packets(), rankreq[0], as_event=True
                        )

                # With > 1 dest_rank, start looping until all dest_rank batches
                # have been sent.
                else:  # if 0 in smd_batch_dict ...
                    # Check if destinations are valid
                    destinations = np.asarray(list(smd_batch_dict.keys()))
                    if any(destinations > n_bd_nodes):
                        self.logger.debug(
                            f"MESSAGE INVALID_DEST ({destinations}). MUST BE <= {n_bd_nodes} (#N_BDS)"
                        )
                        break

                    while smd_batch_dict:
                        if waiting_bds:  # Check first if there are bd nodes waiting
                            copied_waiting_bds = waiting_bds[:]
                            for dest_rank in copied_waiting_bds:
                                if dest_rank in smd_batch_dict:
                                    self._send_to_dest(
                                        dest_rank,
                                        smd_batch_dict,
                                        step_batch_dict,
                                        eb_man,
                                        batches,
                                    )
                                    waiting_bds.remove(dest_rank)

                        if smd_batch_dict:
                            self._request_rank(rankreq)
                            dest_rank = rankreq[0]
                            if dest_rank in smd_batch_dict:
                                self._send_to_dest(
                                    dest_rank,
                                    smd_batch_dict,
                                    step_batch_dict,
                                    eb_man,
                                    batches,
                                )
                            else:
                                waiting_bds.append(dest_rank)
                    # end while smd_batch_dict

                # end else -> if 0 in smd_batch_dict.keys()

                t1 = time.monotonic()
                rate_hz = eb_man.eb.nevents / (t1 - t0)
                self._bd_rate_hz.append(rate_hz)
                # Prometheus metric expects kHz
                self.rate_gauge.set(rate_hz * 1e-3)
                t0 = time.monotonic()

            # end for smd_batch_dict in ...
            self.logger.debug(
                f"TIMELINE 12.1 EB{self.comms.world_rank}DONEALLBATCHES {time.monotonic()}",
            )
            self._log_chunk_stats(chunk_id)
            chunk_id += 1

        # end While True
        if not bypass_bd:
            self._prepare_bd_read_stats(chunk_id - 1, n_bd_nodes)
        wait_for(self.requests)

        batches = {}

        # Check if any of the waiting bds need missing steps from the last batch
        self._init_requests()
        copied_waiting_bds = waiting_bds[:]
        for dest_rank in copied_waiting_bds:
            missing_step_views = self.step_hist.get_buffer(dest_rank)
            batches[dest_rank] = repack_for_bd(
                bytearray(), missing_step_views, self.configs, client=dest_rank
            )
            if batches[dest_rank]:
                self.logger.debug(
                    f"TIMELINE 12.2 EB{self.comms.world_rank}SENDMISSINGSTEPTOBD{dest_rank+1} {time.monotonic()}",
                )
                self.requests[dest_rank - 1] = bd_comm.Isend(
                    batches[dest_rank], dest_rank
                )
                self.logger.debug(
                    f"TIMELINE 12.3 EB{self.comms.world_rank}SENDMISSINGSTEPTOBD{dest_rank+1} {time.monotonic()}",
                )
                waiting_bds.remove(dest_rank)
        wait_for(self.requests)

        self.logger.debug(
            f"TIMELINE 12.4 EB{self.comms.world_rank}DONEMISSSTEPS {time.monotonic()}",
        )

        # Check if the rest of bds need missing steps from the last batch
        self._init_requests()
        for i in range(n_bd_nodes - len(waiting_bds)):
            self._request_rank(rankreq)
            missing_step_views = self.step_hist.get_buffer(rankreq[0])
            batches[rankreq[0]] = repack_for_bd(
                bytearray(), missing_step_views, self.configs, client=rankreq[0]
            )
            if batches[rankreq[0]]:
                self.logger.debug(
                    f"TIMELINE 12.5 EB{self.comms.world_rank}SENDMISSINGSTEPTOBD{rankreq[0]+1} {time.monotonic()}",
                )
                self.requests[rankreq[0] - 1] = bd_comm.Isend(
                    batches[rankreq[0]], dest=rankreq[0]
                )
                self.logger.debug(
                    f"TIMELINE 12.6 EB{self.comms.world_rank}SENDMISSINGSTEPTOBD{rankreq[0]+1} {time.monotonic()}",
                )
            else:
                waiting_bds.append(rankreq[0])
        wait_for(self.requests)

        self.logger.debug(
            f"TIMELINE 12.7 EB{self.comms.world_rank}DONE {time.monotonic()}",
        )

        # end While True: done - kill idling nodes
        self._init_requests()
        for dest_rank in waiting_bds:
            self.requests[dest_rank - 1] = bd_comm.Isend(bytearray(), dest=dest_rank)
        wait_for(self.requests)

        # - kill all other nodes
        self._init_requests()
        for i in range(n_bd_nodes - len(waiting_bds)):
            self._request_rank(rankreq)
            self.requests[rankreq[0] - 1] = bd_comm.Isend(bytearray(), dest=rankreq[0])
        wait_for(self.requests)
        if not bypass_bd:
            bd_stats = self._format_bd_read_stats(self._bd_pending_chunk_id)
            if bd_stats:
                self.logger.debug(bd_stats)

    def start_broadcast(self):
        smd_comm = self.comms.smd_comm
        bd_comm = self.comms.bd_comm

        while True:
            smd_chunk = self._request_data(smd_comm)
            if not smd_chunk:
                break

            eb_man = EventBuilderManager(
                smd_chunk, self.configs, self.dsparms
            )

            for smd_batch_dict, _ in eb_man.batches():
                smd_batch, _ = smd_batch_dict[0]

                # Broadcast to all BD ranks including self
                smd_batch_np = np.frombuffer(smd_batch, dtype='B')
                bd_comm.bcast(smd_batch_np, root=0)

        # Send empty array to signal termination
        bd_comm.bcast(np.array([], dtype='B'), root=0)


class BigDataNode(object):
    def __init__(self, comms, configs, dm, dsparms, shared_state):
        self.comms = comms
        self.configs = configs
        self.dm = dm
        self.dsparms = dsparms
        self.shared_state = shared_state
        pm = get_prom_manager()
        self.wait_gauge = pm.get_metric("psana_bd_wait")
        self.logger = utils.get_logger(name=utils.get_class_name(self))
        self._last_bd_read_bytes = 0
        self._last_bd_read_time_ns = 0
        self._last_bd_wait_time_ns = 0
        self._last_bd_proc_events = 0
        self._last_bd_proc_time_ns = 0

    def start(self):
        def on_batch_end(payload):
            (read_bytes, read_time), event_count, elapsed = payload
            self._last_bd_read_bytes = int(read_bytes)
            self._last_bd_read_time_ns = int(read_time * 1e9)
            self._last_bd_proc_events = int(event_count)
            self._last_bd_proc_time_ns = int(elapsed * 1e9)
            if elapsed > 0 and event_count > 0:
                pass

        def get_smd():
            bd_comm = self.comms.bd_comm
            bd_rank = self.comms.bd_rank
            self.logger.debug(
                f"TIMELINE 13. BD{self.comms.world_rank}SENDREQTOEB {time.monotonic()}",
            )
            req_payload = np.array(
                [
                    bd_rank,
                    self._last_bd_read_bytes,
                    self._last_bd_read_time_ns,
                    self._last_bd_wait_time_ns,
                    self._last_bd_proc_events,
                    self._last_bd_proc_time_ns,
                ],
                dtype=np.int64,
            )
            req = bd_comm.Isend(req_payload, dest=0)
            req.Wait()
            self._last_bd_read_bytes = 0
            self._last_bd_read_time_ns = 0
            self._last_bd_wait_time_ns = 0
            self._last_bd_proc_events = 0
            self._last_bd_proc_time_ns = 0
            self.logger.debug(
                f"TIMELINE 14. BD{self.comms.world_rank}DONESENDREQTOEB {time.monotonic()}",
            )
            info = MPI.Status()
            bd_comm.Probe(source=0, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            chunk = bytearray(count)
            st_req = time.monotonic()
            req = bd_comm.Irecv(chunk, source=0)
            req.Wait()
            self.logger.debug(
                f"TIMELINE 15. BD{self.comms.world_rank}RECVDATA {time.monotonic()}",
            )
            en_req = time.monotonic()
            wait_time = en_req - st_req
            self._last_bd_wait_time_ns = int(wait_time * 1e9)
            self.wait_gauge.set(wait_time)
            return chunk

        dgrams_iter = Events(self.configs,
                        self.dm,
                        self.dsparms.max_retries,
                        self.dsparms.use_smds,
                        self.shared_state,
                        get_smd=get_smd,
                        on_batch_end=on_batch_end)
        for i_evt, dgrams in enumerate(dgrams_iter):
            # throw away events if termination flag is set
            if self.shared_state.terminate_flag.value:
                continue

            yield dgrams

    def start_smdonly(self):
        bd_comm = self.comms.bd_comm

        def get_smd():
            smd_batch_np = bd_comm.bcast(None, root=0)  # receive the broadcast
            count = smd_batch_np.size
            return bytearray(smd_batch_np) if count > 0 else bytearray()

        t0 = time.monotonic()

        events = SmdEvents(self.configs,
                           self.dm,
                           self.dsparms.max_retries,
                           self.dsparms.use_smds,
                           self.shared_state,
                           get_smd=get_smd)
        ts_table = {}

        cn_events = 0
        cn_pass = 0
        def _extract_dgrams_service_ts(item):
            if isinstance(item, (list, tuple)):
                dgrams = item
                svc = utils.first_service(dgrams)
                ts = utils.first_timestamp(dgrams)
                return dgrams, svc, ts

            dgrams = getattr(item, "_dgrams", None)
            if dgrams is None:
                raise TypeError("SmdEvents yielded unsupported type")

            svc_fn = getattr(item, "service", None)
            svc = svc_fn() if callable(svc_fn) else utils.first_service(dgrams)

            ts_attr = getattr(item, "timestamp", None)
            ts = ts_attr() if callable(ts_attr) else ts_attr
            if ts is None:
                ts = utils.first_timestamp(dgrams)

            return dgrams, svc, ts

        for item in events:
            cn_events += 1
            try:
                dgrams, svc, ts = _extract_dgrams_service_ts(item)
            except Exception as exc:
                self.logger.debug(f"Skipping smd entry during table build: {exc}")
                continue

            if svc != TransitionId.L1Accept:
                continue

            ts_table[ts] = {
                i: (d.smdinfo[0].offsetAlg.intOffset, d.smdinfo[0].offsetAlg.intDgramSize)
                for i, d in enumerate(dgrams)
                if d is not None and hasattr(d, 'smdinfo')
            }
            cn_pass += 1

        self.logger.debug(f"build table took {time.monotonic()-t0:.2f}s.")
        return ts_table

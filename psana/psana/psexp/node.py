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
from psana.marchingeventbuilder import MarchingEventBuilder
from psana.psexp.marching_eventmanager import MarchingEventManager

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
    march_shm_comm = None
    march_shm_rank = 0
    march_shm_size = 0
    march_shm_comm = None
    march_shm_rank = 0
    march_shm_size = 0

    def __init__(self):
        self.logger = utils.get_logger(name="Communicators")
        self.comm = MPI.COMM_WORLD
        self.world_rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.world_group = self.comm.Get_group()
        self.hostname = MPI.Get_processor_name()
        self.march_shm_comm = MPI.COMM_NULL

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

        env_march = os.environ.get("PS_MARCHING_READ", "0").strip().lower()
        self.marching_enabled = env_march in ("1", "true", "yes", "on")

        self.bd_main_group = self.psana_group.Excl([0])
        self._bd_only_group = None
        self._srv_group = MPI.Group.Difference(self.world_group, self.psana_group)

        self.smd_group = None
        self.smd_comm = MPI.COMM_NULL
        self.bd_main_comm = self.comm.Create(self.bd_main_group)

        self.march_shared_mem = None
        self.march_params = {}

        if self.bd_main_comm != MPI.COMM_NULL:
            self.bd_main_rank = self.bd_main_comm.Get_rank()
            self.bd_main_size = self.bd_main_comm.Get_size()

            if self.marching_enabled:
                info = MPI.INFO_NULL
                self.march_shm_comm = self.bd_main_comm.Split_type(
                    MPI.COMM_TYPE_SHARED, self.bd_main_rank, info
                )
                if self.march_shm_comm != MPI.COMM_NULL:
                    self.march_shm_rank = self.march_shm_comm.Get_rank()
                    self.march_shm_size = self.march_shm_comm.Get_size()
                    self.bd_comm = self.march_shm_comm
                    self.bd_rank = self.march_shm_rank
                    self.bd_size = self.march_shm_size
                    if self.bd_rank == 0:
                        self._nodetype = "eb"
                    else:
                        self._nodetype = "bd"
            else:
                color = self.bd_main_rank % PS_EB_NODES
                self.bd_comm = self.bd_main_comm.Split(color, self.bd_main_rank)
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

    def bd_group(self):
        return self._bd_only_group

    def srv_group(self):
        return self._srv_group

    def node_type(self):
        return self._nodetype

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
            self.logger.info(
                "[MPI-role] smd_comm size=%d members=[%s]", total, pretty
            )

    def _init_smd_comm(self, ps_eb_nodes, psana_world_size):
        if ps_eb_nodes < 1:
            ps_eb_nodes = 1
        if self.marching_enabled:
            eb_candidate = self.world_rank if self._nodetype == "eb" else -1
            gathered = self.comm.allgather(eb_candidate)
            if self.world_rank == 0:
                eb_worlds = sorted({r for r in gathered if r >= 0})
                if not eb_worlds:
                    raise RuntimeError(
                        "Marching mode requires at least one EB node but none were detected"
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

    def _request_rank(self, rankreq):
        st_req = time.monotonic()
        self.logger.debug(f"TIMELINE 1. SMD0GOTCHUNK {st_req}")

        req = self.comms.smd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
        req.Wait()
        en_req = time.monotonic()
        self.logger.debug(f"TIMELINE 2. SMD0GOTEB{rankreq[0]} {en_req}")
        self.logger.debug(f"WAITTIME SMD0-EB (0-{rankreq[0]}) {en_req-st_req:.5f}")
        self.wait_gauge.set(en_req - st_req)

    def start(self):
        # Rank 0 waits on World comm for terminating signal
        t_rankreq = np.empty(1, dtype="i")
        t_req = self.comms.comm.Irecv(t_rankreq, source=MPI.ANY_SOURCE)

        # Setup a non-pickled recv array and prepare bucket for storing send reqs.
        rankreq = np.empty(1, dtype="i")
        waiting_ebs = []
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]

        # Need this for async MPI to prevent overwriting send buffer
        repack_smds = {}

        # Indentify viewing windows. SmdReaderManager has starting index and block size
        # that it needs to share later when data are packaged for sending to EventBuilders.
        for i_chunk in self.smdr_man.chunks():
            st = time.monotonic()
            self._request_rank(rankreq)

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

            self.logger.debug(f"TIMELINE 4. SMD0DONEWITHEB{rankreq[0]} {time.monotonic()}")

            en = time.monotonic()
            self.logger.debug(
                f"RATE SMD0-EB (0-{rankreq[0]}) {(self.smdr_man.got_events/(en-st))*1e-3} kHz"
            )
            self.rate_gauge.set((self.smdr_man.got_events / (en - st)) * 1e-3)

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

        # end for (smd_chunk, step_chunk)
        wait_for(requests)

        # check if there are missing steps to be sent
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]
        for i in range(self.comms.n_smd_nodes):
            req = self.comms.smd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
            req.Wait()
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
            else:
                waiting_ebs.append(rankreq[0])
        wait_for(requests)

        # kill waiting eb nodes
        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]
        for dest_rank in waiting_ebs:
            requests[dest_rank - 1] = self.comms.smd_comm.Isend(
                bytearray(), dest=dest_rank
            )
        wait_for(requests)

        requests = [MPI.REQUEST_NULL for i in range(self.comms.smd_size - 1)]
        for i in range(self.comms.n_smd_nodes - len(waiting_ebs)):
            req = self.comms.smd_comm.Irecv(rankreq, source=MPI.ANY_SOURCE)
            req.Wait()
            requests[rankreq[0] - 1] = self.comms.smd_comm.Isend(
                bytearray(), dest=rankreq[0]
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
        self.requests = []
        self.logger = utils.get_logger(name=utils.get_class_name(self))

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
        self.logger.debug(
            f"WAITTIME EB-BD ({self.comms.smd_rank}-{rankreq[0]}) {en_req-st_req:.5f}"
        )
        self.wait_bd_gauge.set(en_req - st_req)

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
        self.logger.debug(
            f"WAITTIME EB-SMD0 ({self.comms.smd_rank}-0) {en-st:.5f} ({count/1e3:.5f}KB)"
        )
        self.wait_smd0_gauge.set(en - st)
        return smd_chunk

    def start(self):
        rankreq = np.empty(1, dtype="i")
        smd_comm = self.comms.smd_comm
        n_bd_nodes = self.comms.bd_comm.Get_size() - 1
        bd_comm = self.comms.bd_comm
        waiting_bds = []

        # Initialize Non-blocking Send Requests with Null
        self._init_requests()

        bypass_bd = bool(int(os.environ.get("PS_EB_BYPASS_BD", "0")))

        while True:
            smd_chunk = self._request_data(smd_comm)
            if not smd_chunk:
                break

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
                self.logger.debug(
                    f"RATE EB-BD ({self.comms.smd_rank}-{rankreq[0]}) {(eb_man.eb.nevents/(t1-t0))*1e-3:.5f} kHz"
                )
                self.rate_gauge.set((eb_man.eb.nevents / (t1 - t0)) * 1e-3)
                t0 = time.monotonic()

            # end for smd_batch_dict in ...
            self.logger.debug(
                f"TIMELINE 12.1 EB{self.comms.world_rank}DONEALLBATCHES {time.monotonic()}",
            )

        # end While True
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
            self.logger.debug(f"MESSAGE EB-BD ({self.comms.smd_rank}-{dest_rank}) KILL")
        wait_for(self.requests)

        # - kill all other nodes
        self._init_requests()
        for i in range(n_bd_nodes - len(waiting_bds)):
            self._request_rank(rankreq)
            self.requests[rankreq[0] - 1] = bd_comm.Isend(bytearray(), dest=rankreq[0])
            self.logger.debug(f"MESSAGE EB-BD ({self.comms.smd_rank}-{dest_rank}) KILL")
        wait_for(self.requests)

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


class MarchingEventBuilderNode(object):
    """Feeds marching shared-memory buffers instead of sending MPI batches."""

    def __init__(self, comms, configs, dsparms):
        self.comms = comms
        self.configs = configs
        self.dsparms = dsparms
        self.logger = utils.get_logger(name=utils.get_class_name(self))
        self.shared_mem = getattr(self.comms, "march_shared_mem", None)
        if self.shared_mem is None:
            raise RuntimeError("Marching shared memory is not initialized")
        params = getattr(self.comms, "march_params", {})
        self.builder = MarchingEventBuilder(
            configs,
            dsparms,
            self.shared_mem,
            n_slots=params.get("n_slots", 2),
            max_events_per_chunk=params.get("max_events", dsparms.batch_size),
            max_chunk_bytes=params.get("max_chunk_bytes", 1 << 24),
            name_prefix=params.get("prefix", "march"),
        )

    def _request_data(self):
        smd_comm = self.comms.smd_comm
        smd_comm.Isend(np.array([self.comms.smd_rank], dtype="i"), dest=0)
        info = MPI.Status()
        smd_comm.Probe(source=0, status=info)
        count = info.Get_elements(MPI.BYTE)
        smd_chunk = bytearray(count)
        req = smd_comm.Irecv(smd_chunk, source=0)
        req.Wait()
        return smd_chunk

    def start(self):
        chunk_id = 0
        while True:
            smd_chunk = self._request_data()
            if not smd_chunk:
                # Publish sentinel chunk so BD ranks observe end-of-stream.
                self.builder.publish_shutdown_slot(chunk_id)
                break
            self.builder.ingest_chunk(smd_chunk, chunk_id)
            chunk_id += 1
        self.builder.finalize()



class BigDataNode(object):
    def __init__(self, comms, configs, dm, dsparms, shared_state):
        self.comms = comms
        self.configs = configs
        self.dm = dm
        self.dsparms = dsparms
        self.shared_state = shared_state
        pm = get_prom_manager()
        self.wait_gauge = pm.get_metric("psana_bd_wait")
        self.rate_gauge = pm.get_metric("psana_bd_rate")
        self.logger = utils.get_logger(name=utils.get_class_name(self))

    def start(self):
        def get_smd():
            bd_comm = self.comms.bd_comm
            bd_rank = self.comms.bd_rank
            self.logger.debug(
                f"TIMELINE 13. BD{self.comms.world_rank}SENDREQTOEB {time.monotonic()}",
            )
            req = bd_comm.Isend(np.array([bd_rank], dtype="i"), dest=0)
            req.Wait()
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
            self.logger.debug(
                f"WAITTIME BD-EB ({bd_rank}-{self.comms.smd_rank}) {en_req-st_req:.5f}"
            )
            self.wait_gauge.set(en_req - st_req)
            return chunk

        dgrams_iter = Events(self.configs,
                        self.dm,
                        self.dsparms.max_retries,
                        self.dsparms.use_smds,
                        self.shared_state,
                        get_smd=get_smd)

        t0 = time.monotonic()
        for i_evt, dgrams in enumerate(dgrams_iter):
            # throw away events if termination flag is set
            if self.shared_state.terminate_flag.value:
                continue

            if i_evt % 1000 == 0:
                t1 = time.monotonic()
                rate = 1 / (t1 - t0)
                self.logger.debug(f"RATE BD ({self.comms.bd_rank}-) {rate:.5f} kHz")
                self.rate_gauge.set(rate)
                t0 = time.monotonic()

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
        for evt in events:
            cn_events += 1
            if evt.service() != TransitionId.L1Accept:
                continue

            ts = evt.timestamp
            ts_table[ts] = {
                i: (d.smdinfo[0].offsetAlg.intOffset, d.smdinfo[0].offsetAlg.intDgramSize)
                for i, d in enumerate(evt._dgrams)
                if d is not None and hasattr(d, 'smdinfo')
            }
            cn_pass += 1

        self.logger.debug(f"build table took {time.monotonic()-t0:.2f}s.")
        return ts_table


class MarchingBigDataNode(object):
    """Consumes marching shared-memory slots and yields events."""

    def __init__(self, comms, configs, dm, dsparms, shared_state):
        self.comms = comms
        self.configs = configs
        self.dm = dm
        self.dsparms = dsparms
        self.shared_state = shared_state
        self.shared_mem = getattr(self.comms, "march_shared_mem", None)
        if self.shared_mem is None:
            raise RuntimeError("Marching shared memory is not initialized")
        pm = get_prom_manager()
        self.wait_gauge = pm.get_metric("psana_bd_wait")
        self.rate_gauge = pm.get_metric("psana_bd_rate")
        self.logger = utils.get_logger(name=utils.get_class_name(self))

    def start(self):
        use_prange_env = os.environ.get("PS_PREAD_USE_PRANGE", "1").lower()
        if use_prange_env in ("0", "false", "off"):
            bd_rank = getattr(self.comms, "bd_rank", None)
            if bd_rank == 1 and os.environ.get("PS_PREAD_PRANGE_WARNED", "0") != "1":
                self.logger.warning(
                    "PS_PREAD_USE_PRANGE disabled; marching pread calls will run serialized"
                )
                os.environ["PS_PREAD_PRANGE_WARNED"] = "1"
        params = getattr(self.comms, "march_params", {})
        n_consumers = max(self.comms.march_shm_size - 1, 1)
        evt_mgr = MarchingEventManager(
            self.configs,
            self.dm,
            self.shared_mem,
            n_consumers=n_consumers,
            shared_state=self.shared_state,
            name_prefix=params.get("prefix", "march"),
            use_smds=self.dsparms.use_smds,
            events_per_grant=getattr(self.dsparms, "march_events_per_grant", 1),
        )
        t0 = time.monotonic()
        for i_evt, dgrams in enumerate(evt_mgr):
            if self.shared_state.terminate_flag.value:
                continue
            if i_evt and i_evt % 1000 == 0:
                t1 = time.monotonic()
                rate = 1000 / (t1 - t0)
                self.logger.debug(
                    f"RATE MARCH BD ({self.comms.world_rank}) {rate:.3f} Hz"
                )
                self.rate_gauge.set(rate)
                t0 = time.monotonic()
            yield dgrams

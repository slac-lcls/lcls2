import os
import time

import numpy as np

from psana import dgram, utils
from psana.psexp.transitionid import TransitionId
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.tools import mode
from psana.psexp.prometheus_manager import get_prom_manager
from psana.psexp import bd_plan

if mode == "mpi":
    pass


class ExitId:
    NoError = 0
    BdReadFail = 1


class EventManager(object):
    """Return an event from the received smalldata memoryview (view)

    1) If dm is empty (no bigdata), yield this smd event
    2) If dm is not empty,
        - with filter fn, fetch one bigdata and yield it.
        - w/o filter fn, fetch one big chunk of bigdata and
          replace smalldata view with the read out bigdata.
          Yield one bigdata event.
    """

    def __init__(
        self,
        view,
        configs,
        dm,
        max_retries,
        use_smds,
        smd=False,
        bd_plan=None,
    ):
        if view:
            pf = PacketFooter(view=view)
            self.n_events = pf.n_packets
        else:
            self.n_events = 0

        self.smd_view = view
        self.smd_configs = configs
        self.dm = dm
        self.n_smd_files = len(self.smd_configs)
        self.read_gauge = get_prom_manager().get_metric("psana_bd_read")
        self.max_retries = max_retries
        self.use_smds = use_smds
        self.i_evt = 0
        self.exit_id = ExitId.NoError
        self.smd_mode = smd
        self.bd_plan = bd_plan

        self.logger = utils.get_logger(name=utils.get_class_name(self))

        # Store chunkid and chunk filename
        self.chunkinfo = {}

        # Each chunk must fit in BD_CHUNKSIZE and we only fill bd buffers
        # when bd_offset reaches the size of buffer.
        self.BD_CHUNKSIZE = int(os.environ.get("PS_BD_CHUNKSIZE", 0x1000000))
        self._get_offset_and_size()
        self.plan_chunks_by_file = None
        if self.bd_plan:
            self._init_plan_state()
        if self.dm.n_files > 0:
            self._init_bd_chunks()
            self._preview_bd_plan()

    def __iter__(self):
        return self

    def __next__(self):
        # Check in case there are some failures (I/O) happened on a core.
        # For MPI Mode, this allows clean exit.
        if self.exit_id > 0:
            raise StopIteration

        if self.i_evt == self.n_events:
            raise StopIteration

        dgrams = self._get_next_dgrams()

        if dgrams is None:
            raise StopIteration

        return dgrams

    def isEvent(self, service):
        """EventManager event is considered as:
        - TransitionId.isEvent()
        - service is 0, which only happens to only L1Accept types when the dgram is
          missing from that stream. The service is marked as 0.
        """
        if TransitionId.isEvent(service) or service == 0:
            return True
        else:
            return False

    def _get_offset_and_size(self):
        """
        Use fast step-through to read off offset and size from smd_view.
        Format of smd_view
        [
          [[d_bytes][d_bytes]....[evt_footer]] <-- 1 event
          [[d_bytes][d_bytes]....[evt_footer]]
          [chunk_footer]]
        """
        chunk_id_cb = None
        if self.dm and self.dm.n_files > 0:
            chunk_id_cb = self.dm.get_chunk_id

        offsets = bd_plan.compute_offset_tables(
            self.smd_view,
            self.smd_configs,
            self.use_smds,
            self.BD_CHUNKSIZE,
            chunk_id_cb=chunk_id_cb,
        )

        self.bd_offset_array = offsets.bd_offset_array
        self.bd_size_array = offsets.bd_size_array
        self.smd_offset_array = offsets.smd_offset_array
        self.smd_size_array = offsets.smd_size_array
        self.new_chunk_id_array = offsets.new_chunk_id_array
        self.service_array = offsets.service_array
        self.cutoff_indices = offsets.cutoff_indices
        self.chunk_indices = np.zeros(self.n_smd_files, dtype=np.int64)
        self.chunkinfo.update(offsets.chunkinfo)

    def _open_new_bd_file(self, i_smd, new_chunk_id):
        os.close(self.dm.fds[i_smd])
        xtc_dir = os.path.dirname(self.dm.xtc_files[i_smd])
        new_filename = os.path.join(xtc_dir, self.chunkinfo[(i_smd, new_chunk_id)])
        fd = os.open(new_filename, os.O_RDONLY)
        self.dm.fds[i_smd] = fd
        self.dm.xtc_files[i_smd] = new_filename
        self.dm.set_chunk_id(i_smd, new_chunk_id)

    def _read(self, fd, size, offset):
        st = time.monotonic()
        chunk = bytearray()

        request_size = size
        for i_retry in range(self.max_retries + 1):
            new_read = os.pread(fd, size, offset)
            chunk.extend(new_read)
            got = memoryview(new_read).nbytes
            if memoryview(chunk).nbytes == request_size:
                break

            # Check if we should exit when asked amount is not fulfilled
            if i_retry == self.max_retries and got < size:
                if self.max_retries > 0:
                    # Live mode use max_retries
                    print("Error: maximum no. of retries reached. Exit.")
                else:
                    # Normal mode
                    print(
                        f"Error: not able to completely read big data (asked: {size} bytes/ got: {got} bytes)"
                    )

                # Flag failure for system exit
                self.exit_id = ExitId.BdReadFail
                break

            offset += got
            size -= got

            self.logger.warning(
                f"bigdata read retry#{i_retry}/{self.max_retries} {self.dm.fds_map[fd]} ask={size} offset={offset} got={got}"
            )

            time.sleep(1)

        en = time.monotonic()
        sum_read_nbytes = memoryview(chunk).nbytes  # for prometheus counter
        if sum_read_nbytes > 0:
            rate = (sum_read_nbytes / 1e6) / (en - st)
            self.logger.debug(
                f"bd reads chunk {sum_read_nbytes/1e6:.5f} MB took {en-st:.2f} s (Rate: {rate:.2f} MB/s)"
            )
            self.read_gauge.set(rate)
        return chunk

    def _init_bd_chunks(self):
        self.bd_bufs = [bytearray() for i in range(self.n_smd_files)]
        self.bd_buf_offsets = np.zeros(self.n_smd_files, dtype=np.int64)

    def _preview_bd_plan(self):
        """
        Log how current chunk reads are interleaved vs. a per-file sequential order.
        This is only for debugging/prototyping the new EB batching strategy.
        """
        if not int(os.environ.get("PS_BD_PREVIEW_PLAN", "0")):
            return

        plan = self.bd_plan or bd_plan.plan_from_event_manager(self)
        if not plan:
            return
        descriptors = plan["chunks"]
        if not descriptors:
            return

        # Current behaviour: chunks are consumed in event order and we read whatever
        # file hits its cutoff next (event index + file index tie-breaker).
        current_seq = sorted(
            descriptors, key=lambda d: (d["start_evt"], d["file_index"])
        )
        # Proposed behaviour: drain one file at a time so that each BD issues large
        # sequential reads from a single xtc file before switching.
        per_file_seq = sorted(
            descriptors, key=lambda d: (d["file_index"], d["chunk_index"])
        )

        limit = int(os.environ.get("PS_BD_PREVIEW_LIMIT", "12"))

        def _fmt(seq):
            lines = []
            for entry in seq[:limit]:
                lines.append(
                    "file={file_name} chunk={chunk_index} evt={start_evt}->{end_evt} "
                    "n_offsets={n_offsets} bytes={total_bytes} start={start_offset}".format(
                        **entry
                    )
                )
            if len(seq) > limit:
                lines.append(f"... ({len(seq) - limit} more chunks)")
            return lines

        current_lines = "\n    ".join(_fmt(current_seq))
        target_lines = "\n    ".join(_fmt(per_file_seq))
        self.logger.info(
            "BD chunk preview (current interleaved order):\n    %s", current_lines
        )
        self.logger.info(
            "BD chunk preview (file-by-file order):\n    %s", target_lines
        )

    def _bd_filename(self, i_smd):
        if hasattr(self.dm, "xtc_files") and i_smd < len(self.dm.xtc_files):
            return os.path.basename(self.dm.xtc_files[i_smd])
        return f"smd-{i_smd}"

    def _init_plan_state(self):
        plan = self.bd_plan
        if not plan:
            return
        chunks = plan.get("chunks", [])
        if not chunks:
            return
        per_file = [[] for _ in range(self.n_smd_files)]
        for entry in chunks:
            idx = entry.get("file_index")
            if idx is None or idx >= self.n_smd_files:
                continue
            per_file[idx].append(entry)
        # sort by chunk_index to keep stable order
        for lst in per_file:
            lst.sort(key=lambda e: e.get("chunk_index", 0))
        self.plan_chunks_by_file = per_file

    def _fill_bd_chunk(self, i_smd):
        """
        Fill self.bigdatas for this given stream id
        No filling: No bigdata files or
           when this stream doesn't have at least a dgram.
        Detail:
            - Ignore all transitions
            - Read the next chunk
              From cutoff_flag_array[:, i_smd], we get cutoff_indices as
              [0, 1, 2, 12, 13, 18, 19] a list of index to where each read
              or copy will happen.
        """
        # Check no filling
        if self.use_smds[i_smd]:
            return

        # Reset buffer offset with new filling
        self.bd_buf_offsets[i_smd] = 0

        entry = None
        if self.plan_chunks_by_file:
            file_chunks = self.plan_chunks_by_file[i_smd]
            idx = self.chunk_indices[i_smd]
            if idx < len(file_chunks):
                entry = file_chunks[idx]

        if entry:
            begin_chunk_offset = entry["start_offset"]
            read_size = entry["total_bytes"]
        else:
            cutoff_indices = self.cutoff_indices[i_smd]
            i_evt_cutoff = cutoff_indices[self.chunk_indices[i_smd]]
            begin_chunk_offset = self.bd_offset_array[i_evt_cutoff, i_smd]
            # Calculate read size:
            if self.chunk_indices[i_smd] == cutoff_indices.shape[0] - 1:
                read_size = np.sum(self.bd_size_array[i_evt_cutoff:, i_smd])
            else:
                i_next_evt_cutoff = cutoff_indices[self.chunk_indices[i_smd] + 1]
                read_size = np.sum(
                    self.bd_size_array[i_evt_cutoff:i_next_evt_cutoff, i_smd]
                )

        self.bd_bufs[i_smd] = self._read(
            self.dm.fds[i_smd], read_size, begin_chunk_offset
        )

    def _get_next_dgrams(self):
        """Generate bd evt for different cases:
        1) No bigdata or Transition Event
            create dgrams from smd_view
        2) L1Accept event
            create dgrams from bd_bufs
        3) L1Accept with some smd files replaced by bigdata files
            create dgram from smd_view if use_smds[i_smd] is set
            otherwise create dgram from bd_bufs
        """
        dgrams = [None] * self.n_smd_files
        for i_smd in range(self.n_smd_files):
            if (
                self.dm.n_files == 0
                or not self.isEvent(self.service_array[self.i_evt, i_smd])
                or self.use_smds[i_smd]
                or self.smd_mode
            ):
                view = self.smd_view
                offset = self.smd_offset_array[self.i_evt, i_smd]
                size = self.smd_size_array[self.i_evt, i_smd]

                # Non L1 always are counted as a new "chunk" since they
                # ther cutoff flag is set (data coming from smd view
                # instead of bd chunk. We'll need to update chunk index for
                # this smd when we see non L1.
                self.chunk_indices[i_smd] += 1

                # Check in case we need to switch to the next bigdata chunk file
                if not self.isEvent(self.service_array[self.i_evt, i_smd]):
                    if self.new_chunk_id_array[self.i_evt, i_smd] != 0:
                        self._open_new_bd_file(
                            i_smd, self.new_chunk_id_array[self.i_evt, i_smd]
                        )
            else:
                # Fill up bd buf if this dgram doesn't fit in the current view
                if (
                    self.bd_buf_offsets[i_smd] + self.bd_size_array[self.i_evt, i_smd]
                    > memoryview(self.bd_bufs[i_smd]).nbytes
                ):
                    # Guard: Only fill chunk if cutoff exists
                    if self.chunk_indices[i_smd] >= len(self.cutoff_indices[i_smd]):
                        return None
                    else:
                        self._fill_bd_chunk(i_smd)
                        self.chunk_indices[i_smd] += 1

                # This is the offset of bd buffer! and not what stored in smd dgram,
                # which in contrast points to the location of disk.
                offset = self.bd_buf_offsets[i_smd]
                size = self.bd_size_array[self.i_evt, i_smd]
                view = self.bd_bufs[i_smd]
                self.bd_buf_offsets[i_smd] += size

            if size > 0:  # handles missing dgram
                dgrams[i_smd] = dgram.Dgram(
                    config=self.dm.configs[i_smd], view=view, offset=offset
                )
                if (
                    self.service_array[self.i_evt, i_smd]
                    == TransitionId.L1Accept_EndOfBatch
                ):
                    setattr(dgrams[i_smd], "_endofbatch", True)

        self.i_evt += 1
        return dgrams

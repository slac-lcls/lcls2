import os
import time

import numpy as np

from psana import dgram, utils
from psana.psexp import TransitionId
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.tools import mode

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
        self.max_retries = max_retries
        self.use_smds = use_smds
        self.i_evt = 0
        self.exit_id = ExitId.NoError
        self.smd_mode = smd

        self.logger = utils.get_logger(name=utils.get_class_name(self))
        self._bd_read_bytes = 0
        self._bd_read_time = 0.0

        # Store chunkid and chunk filename
        self.chunkinfo = {}

        # Each chunk must fit in BD_CHUNKSIZE and we only fill bd buffers
        # when bd_offset reaches the size of buffer.
        self.BD_CHUNKSIZE = int(os.environ.get("PS_BD_CHUNKSIZE", 0x1000000))
        self._get_offset_and_size()
        if self.dm.n_files > 0:
            self._init_bd_chunks()

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

    def _get_bd_offset_and_size(
        self, d, current_bd_offsets, current_bd_chunk_sizes, i_evt, i_smd, i_first_L1
    ):
        if self.use_smds[i_smd]:
            return

        self.bd_offset_array[i_evt, i_smd] = d.smdinfo[0].offsetAlg.intOffset
        self.bd_size_array[i_evt, i_smd] = d.smdinfo[0].offsetAlg.intDgramSize

        # Check continuous chunk
        if (
            current_bd_offsets[i_smd] == self.bd_offset_array[i_evt, i_smd]
            and i_evt != i_first_L1
            and current_bd_chunk_sizes[i_smd] + self.bd_size_array[i_evt, i_smd]
            < self.BD_CHUNKSIZE
        ):
            self.cutoff_flag_array[i_evt, i_smd] = 0
            current_bd_chunk_sizes[i_smd] += self.bd_size_array[i_evt, i_smd]
        else:
            current_bd_chunk_sizes[i_smd] = self.bd_size_array[i_evt, i_smd]

        current_bd_offsets[i_smd] = (
            self.bd_offset_array[i_evt, i_smd] + self.bd_size_array[i_evt, i_smd]
        )

    def _get_offset_and_size(self):
        """
        Use fast step-through to read off offset and size from smd_view.
        Format of smd_view
        [
          [[d_bytes][d_bytes]....[evt_footer]] <-- 1 event
          [[d_bytes][d_bytes]....[evt_footer]]
          [chunk_footer]]
        """
        offset = 0
        i_smd = 0
        smd_chunk_pf = PacketFooter(view=self.smd_view)
        dtype = np.int64
        # Row - events, col = smd files
        self.bd_offset_array = np.zeros(
            (smd_chunk_pf.n_packets, self.n_smd_files), dtype=dtype
        )
        self.bd_size_array = np.zeros(
            (smd_chunk_pf.n_packets, self.n_smd_files), dtype=dtype
        )
        self.smd_offset_array = np.zeros(
            (smd_chunk_pf.n_packets, self.n_smd_files), dtype=dtype
        )
        self.smd_size_array = np.zeros(
            (smd_chunk_pf.n_packets, self.n_smd_files), dtype=dtype
        )
        self.new_chunk_id_array = np.zeros(
            (smd_chunk_pf.n_packets, self.n_smd_files), dtype=dtype
        )
        self.cutoff_flag_array = np.ones(
            (smd_chunk_pf.n_packets, self.n_smd_files), dtype=dtype
        )
        self.service_array = np.zeros(
            (smd_chunk_pf.n_packets, self.n_smd_files), dtype=dtype
        )

        smd_aux_sizes = np.zeros(self.n_smd_files, dtype=dtype)
        # For comparing if the next dgram should be in the same read
        current_bd_offsets = np.zeros(self.n_smd_files, dtype=dtype)
        # Current chunk size (gets reset at boundary)
        current_bd_chunk_sizes = np.zeros(self.n_smd_files, dtype=dtype)
        i_evt = 0
        i_first_L1 = -1
        while (
            offset
            < memoryview(self.smd_view).nbytes - memoryview(smd_chunk_pf.footer).nbytes
        ):
            if i_smd == 0:
                smd_evt_size = smd_chunk_pf.get_size(i_evt)
                smd_evt_pf = PacketFooter(
                    view=self.smd_view[offset : offset + smd_evt_size]
                )
                smd_aux_sizes[:] = [
                    smd_evt_pf.get_size(i) for i in range(smd_evt_pf.n_packets)
                ]

            # Only get offset and size of non-missing dgram
            # TODO: further optimization by looking for the first L1 and read in a big chunk
            # anything that comes after. Right now, all transitions mark the cutoff points.
            if smd_aux_sizes[i_smd] == 0:
                self.cutoff_flag_array[i_evt, i_smd] = 0
            else:
                d = dgram.Dgram(
                    config=self.smd_configs[i_smd], view=self.smd_view, offset=offset
                )

                self.smd_offset_array[i_evt, i_smd] = offset
                self.smd_size_array[i_evt, i_smd] = d._size
                self.service_array[i_evt, i_smd] = d.service()

                # For L1 with bigdata files, store offset and size found in smd dgrams.
                # For Enable, store new chunk id (if found).
                if self.isEvent(d.service()) and self.dm.n_files > 0:
                    if i_first_L1 == -1:
                        i_first_L1 = i_evt
                    self._get_bd_offset_and_size(
                        d,
                        current_bd_offsets,
                        current_bd_chunk_sizes,
                        i_evt,
                        i_smd,
                        i_first_L1,
                    )
                elif d.service() == TransitionId.Enable and hasattr(d, "chunkinfo"):
                    # We only support chunking on bigdata
                    if self.dm.n_files > 0:
                        _chunk_ids = [
                            getattr(d.chunkinfo[seg_id].chunkinfo, "chunkid")
                            for seg_id in d.chunkinfo
                        ]
                        _chunk_filenames = [
                            getattr(d.chunkinfo[seg_id].chunkinfo, "filename")
                            for seg_id in d.chunkinfo
                        ]
                        # Only flag new chunk when there's chunkinfo and that chunkid is new
                        if _chunk_ids:
                            # There must be only one unique chunkid name
                            new_chunk_id = _chunk_ids[0]
                            new_filename = _chunk_filenames[0]
                            current_chunk_id = self.dm.get_chunk_id(i_smd)
                            if new_chunk_id > current_chunk_id:
                                self.new_chunk_id_array[i_evt, i_smd] = new_chunk_id
                                self.chunkinfo[(i_smd, new_chunk_id)] = new_filename

            offset += smd_aux_sizes[i_smd]
            i_smd += 1
            if i_smd == self.n_smd_files:
                offset += PacketFooter.n_bytes * (
                    self.n_smd_files + 1
                )  # skip the footer
                i_smd = 0  # reset to the first smd file
                i_evt += 1  # done with this smd event

        # end while offset

        # Precalculate cutoff indices
        self.cutoff_indices = []
        self.chunk_indices = np.zeros(self.n_smd_files, dtype=dtype)
        for i_smd in range(self.n_smd_files):
            self.cutoff_indices.append(
                np.where(self.cutoff_flag_array[:, i_smd] == 1)[0]
            )

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
            elapsed = en - st
            if elapsed > 0:
                self._bd_read_bytes += sum_read_nbytes
                self._bd_read_time += elapsed
        return chunk

    def get_bd_read_stats(self):
        return int(self._bd_read_bytes), float(self._bd_read_time)

    def _init_bd_chunks(self):
        self.bd_bufs = [bytearray() for i in range(self.n_smd_files)]
        self.bd_buf_offsets = np.zeros(self.n_smd_files, dtype=np.int64)

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

        cutoff_indices = self.cutoff_indices[i_smd]
        i_evt_cutoff = cutoff_indices[self.chunk_indices[i_smd]]
        begin_chunk_offset = self.bd_offset_array[i_evt_cutoff, i_smd]

        # Calculate read size:
        # For last chunk, read size is the sum of all bd dgrams all the
        # way to the end of the array. Otherwise, only sum to the next chunk.
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

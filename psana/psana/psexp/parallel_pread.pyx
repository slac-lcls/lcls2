# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.string cimport strerror
from libc.errno cimport errno
from cpython.buffer cimport PyBUF_READ
from time import perf_counter
import os

ctypedef long off_t
ctypedef long ssize_t

cdef extern from "unistd.h":
    ssize_t pread(int fd, void* buf, size_t count, off_t offset) nogil

cdef extern from "Python.h":
    object PyMemoryView_FromMemory(char *mem, Py_ssize_t size, int flags)


cdef double _total_pread_seconds = 0
cdef long long _total_pread_bytes = 0
cdef long long _total_pread_calls = 0

cpdef tuple parallel_pread_stats():
    return (_total_pread_seconds, _total_pread_bytes, _total_pread_calls)

cpdef void reset_parallel_pread_stats():
    global _total_pread_seconds, _total_pread_bytes, _total_pread_calls
    _total_pread_seconds = 0
    _total_pread_bytes = 0
    _total_pread_calls = 0


cdef class ParallelPreader:
    """
    Perform concurrent pread() calls across multiple file descriptors.
    Buffers are allocated once per stream and reused for every read.
    """

    cdef int n_streams
    cdef size_t* max_sizes
    cdef unsigned char** buffers

    def __cinit__(self, int n_streams, list max_sizes):
        if n_streams <= 0:
            raise ValueError("n_streams must be > 0")
        if len(max_sizes) != n_streams:
            raise ValueError("max_sizes must match n_streams")
        self.n_streams = n_streams
        self.max_sizes = <size_t*> malloc(n_streams * sizeof(size_t))
        if self.max_sizes is NULL:
            raise MemoryError()
        self.buffers = <unsigned char**> malloc(n_streams * sizeof(unsigned char*))
        if self.buffers is NULL:
            free(self.max_sizes)
            raise MemoryError()
        cdef int i
        cdef size_t sz
        for i in range(n_streams):
            sz = <size_t> max_sizes[i]
            if sz == 0:
                sz = 1
            self.max_sizes[i] = sz
            self.buffers[i] = <unsigned char*> malloc(sz)
            if self.buffers[i] is NULL:
                for i in range(n_streams):
                    if self.buffers[i] != NULL:
                        free(self.buffers[i])
                free(self.buffers)
                free(self.max_sizes)
                raise MemoryError()

    def __dealloc__(self):
        cdef int i
        if self.buffers is not NULL:
            for i in range(self.n_streams):
                if self.buffers[i] is not NULL:
                    free(self.buffers[i])
            free(self.buffers)
        if self.max_sizes is not NULL:
            free(self.max_sizes)

    cpdef list read(self, int[:] fds, long long[:] offsets, Py_ssize_t[:] sizes):
        """
        Launch concurrent pread calls for each (fd, offset, size) triple.
        Returns a list of memoryviews referencing the internal buffers filled
        with the requested data. Buffers are overwritten on each call.
        """
        global _total_pread_seconds, _total_pread_bytes, _total_pread_calls
        cdef Py_ssize_t n = fds.shape[0]
        if n != self.n_streams or offsets.shape[0] != n or sizes.shape[0] != n:
            raise ValueError("fds/offsets/sizes must all match n_streams")

        cdef Py_ssize_t* read_counts = <Py_ssize_t*> malloc(n * sizeof(Py_ssize_t))
        cdef int* errcodes = <int*> malloc(n * sizeof(int))
        if read_counts is NULL or errcodes is NULL:
            if read_counts != NULL:
                free(read_counts)
            if errcodes != NULL:
                free(errcodes)
            raise MemoryError()

        cdef Py_ssize_t i
        cdef double t0 = perf_counter()
        if os.environ.get("PS_PREAD_USE_PRANGE", "0") not in ("0", "false", "False"):
            with nogil:
                for i in prange(n, schedule='static'):
                    errcodes[i] = 0
                    if sizes[i] < 0 or <size_t>sizes[i] > self.max_sizes[i]:
                        errcodes[i] = -2  # size error
                        read_counts[i] = -1
                        continue
                    read_counts[i] = pread(fds[i],
                                           <void*> self.buffers[i],
                                           <size_t> sizes[i],
                                           offsets[i])
                    if read_counts[i] != sizes[i]:
                        if read_counts[i] < 0:
                            errcodes[i] = errno
                        else:
                            errcodes[i] = -1  # short read
        else:
            with nogil:
                for i in range(n):
                    errcodes[i] = 0
                    if sizes[i] < 0 or <size_t>sizes[i] > self.max_sizes[i]:
                        errcodes[i] = -2  # size error
                        read_counts[i] = -1
                        continue
                    read_counts[i] = pread(fds[i],
                                           <void*> self.buffers[i],
                                           <size_t> sizes[i],
                                           offsets[i])
                    if read_counts[i] != sizes[i]:
                        if read_counts[i] < 0:
                            errcodes[i] = errno
                        else:
                            errcodes[i] = -1  # short read

        cdef double elapsed = perf_counter() - t0
        
        cdef list views = []
        cdef Py_ssize_t bytes_read
        cdef int err
        cdef int ok = 1
        cdef int idx
        cdef long long interval_bytes = 0
        for idx in range(n):
            err = errcodes[idx]
            bytes_read = read_counts[idx]
            if err != 0 or bytes_read < 0:
                ok = 0
                break
            mv = PyMemoryView_FromMemory(<char*> self.buffers[idx],
                                         sizes[idx],
                                         PyBUF_READ)
            views.append(mv)
            interval_bytes += <long long> sizes[idx]

        free(read_counts)
        free(errcodes)

        if not ok:
            if err == -2:
                raise ValueError(f"Requested size {sizes[idx]} exceeds buffer capacity")
            elif err == -1:
                raise IOError(f"Short read for stream {idx}: expected {sizes[idx]}, got {bytes_read}")
            else:
                raise OSError(err, strerror(err))

        global _total_pread_seconds, _total_pread_bytes, _total_pread_calls
        _total_pread_seconds += elapsed
        _total_pread_bytes += interval_bytes
        _total_pread_calls += 1

        return views

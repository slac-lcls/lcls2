# psana/mman_compat.pxd
# POSIX mmap/munmap declarations shared by ParallelReader and SmdReader.
# -----------------------------------------------------------------------------
# Platform notes (Linux vs macOS) for mmap/munmap declarations
#
# • This .pxd cimports POSIX memory-mapping symbols from <sys/mman.h>.
#   On Linux (glibc), the following are available and behave as expected:
#     - mmap, munmap
#     - PROT_READ, PROT_WRITE
#     - MAP_PRIVATE, MAP_ANONYMOUS  (aka MAP_ANON on some systems)
#   Optional/extra flags such as MAP_POPULATE or MADV_* (madvise) are Linux-specific
#   and are NOT declared here to keep this file portable.
#
# • On macOS (Darwin):
#     - <sys/mman.h> exists and mmap/munmap work.
#     - MAP_ANONYMOUS may be defined as MAP_ANON. If your compiler only exposes
#       MAP_ANON, you can alias it at compile time (see snippet below).
#     - Linux-only flags like MAP_POPULATE and some MADV_* constants do not exist.
#     - glibc’s malloc_trim(0) does not exist on macOS (different malloc).
#
# • Recommended portability pattern:
#     - Keep this .pxd strictly to the common POSIX subset (mmap/munmap + basic flags).
#     - Do any OS-specific aliases in the .pyx using a compile-time macro from setup.py.
#
#   Example alias in the .pyx (define PSANA_DARWIN=1 in setup.py when building on macOS):
#
#       IF DEF PSANA_DARWIN:
#           cdef int MAP_ANON_OR_ANONYM = MAP_ANON
#       ELSE:
#           cdef int MAP_ANON_OR_ANONYM = MAP_ANONYMOUS
#
#       # Then use MAP_ANON_OR_ANONYM in calls instead of MAP_ANONYMOUS.
# -----------------------------------------------------------------------------

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

cdef extern from "sys/mman.h":
    void *mmap(void *addr, size_t length, int prot, int flags, int fd, uintptr_t offset)
    int   munmap(void *addr, size_t length)
    int   PROT_READ
    int   PROT_WRITE
    int   MAP_PRIVATE
    int   MAP_ANONYMOUS  # On macOS may be MAP_ANON; alias in .pyx if needed
    void *MAP_FAILED

cdef inline bint is_map_failed(void* p) nogil:
    return p == MAP_FAILED

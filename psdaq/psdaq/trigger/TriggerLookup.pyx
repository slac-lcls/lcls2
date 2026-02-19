from libcpp.vector cimport vector
import EbDgram as edg

cdef vector[char*] get_teb_lookup(event):
     cdef vector[char*] p
     cdef int i, n
     n = len(event._shm_bufSizes)
     p.resize(len(event._det_src))
     for i in range(n):
         if (event._ctrb >> i)&1:
            beg = event._shm_bufSizes[i]
            end = event._shm_bufSizes[i + 1]
            datagram = edg.EbDgram(view=event._shm_inp_mmap[beg:end])
            src = datagram.xtc.src.value()
            if src in event._det_src:
               p[ event._det_src[src] ] = datagram.xtc.payload()

     return p

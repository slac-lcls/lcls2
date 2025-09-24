
from libcpp cimport bool

cdef extern from "psalg/shmem/ShmemClient.hh" namespace "psalg::shmem":
    cdef cppclass ShmemClient:
        int connect(const char* tag, int tr_index)
        # Mark nogil so GIL can be released when calling into them
        # Otherwise have to link ShmemClient to Python and release GIL there
        # Release required for multi-threaded dgrammanager implementation
        void *get(int& ev_index, size_t& buf_size, bool& eventSkipped, bool transitionsOnly) nogil
        void free(int ev_index, size_t buf_size) nogil

cdef class PyShmemClient:
    """ Python wrapper for C++ class.
    """
    cdef ShmemClient* client  # holds a C++ pointer to instance
    cdef char[:] pclient           # used to pass C++ pointer to python

    def __cinit__(self):
        self.client = new ShmemClient()
        self.pclient = <char[:sizeof(ShmemClient)]><char*>self.client

    def __dealloc__(self):
        del self.client

    def connect(self, tag, tr_index):
        # GIL fine here currently as method is called from main thread
        cdef int status = -1
        status = self.client.connect(tag.encode(),tr_index)
        return status

    def get(self,args):
        cdef char* buf
        cdef char[:] cview
        cdef int ev_index = -1
        cdef size_t buf_size = 0

        # Coerece bool
        cdef bool eventSkipped = args['eventSkipped']
        cdef bool transitionsOnly = args['transitionsOnly']

        with nogil:
            buf = <char*>self.client.get(ev_index,buf_size,eventSkipped,transitionsOnly)
        # Need to update this reference before returning None
        args['eventSkipped'] = eventSkipped
        if buf == NULL:
          return

        #this needs to be done with kwargs
        args['index'] = ev_index
        args['size'] = buf_size
        args['cli_cptr'] = self.pclient

        cview = <char[:buf_size]>buf

        return cview

    def free(self,dgram):
        # Need to coerce values before releasing GIL since coercion could throw
        # an exception
        cdef int idx = dgram._shmem_index
        cdef size_t size = dgram._shmem_size
        with nogil:
            self.client.free(idx, size)

    def freeByIndex(self, index, size):
        # Need to coerce values before releasing GIL since coercion could throw
        # an exception
        cdef int idx = index
        cdef size_t c_size = size
        with nogil:
            self.client.free(idx, c_size)

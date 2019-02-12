
cdef extern from "psalg/shmem/ShmemClient.hh" namespace "psalg::shmem":
    cdef cppclass ShmemClient:
        void connect(const char* tag, int tr_index)
        void *get(int& ev_index, int& buf_size)
        void free(int ev_index, int buf_size)

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
        self.client.connect(tag.encode(),tr_index)
    
    def get(self,args):
        cdef char* buf
        cdef char[:] cview
        cdef int ev_index = -1
        cdef int buf_size = 0
                
        buf = <char*>self.client.get(ev_index,buf_size)
        if buf == NULL:
          return

        #this needs to be done with kwargs
        args[0] = ev_index
        args[1] = buf_size
        args[2] = self.pclient

        cview = <char[:buf_size]>buf
        
        return cview
                 
    def free(self,dgram):
        self.client.free(dgram._shmem_index,dgram._shmem_size)

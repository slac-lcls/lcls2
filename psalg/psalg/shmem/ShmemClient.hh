#ifndef PsAlg_ShMem_ShmemClient_hh
#define PsAlg_ShMem_ShmemClient_hh

#include <poll.h>

namespace XtcData {

    class Dgram;

};

namespace psalg {
  namespace shmem {

    class DgramHandler;
    
    class ShmemClient {
    public:
      ShmemClient() {}
      virtual ~ShmemClient() {};

    public:
      //
      //  tr_index must be unique among clients
      //  unique values of ev_index produce a serial chain of clients sharing events
      //  common values of ev_index produce a set of clients competing for events
      //
      int connect(const char* tag, int tr_index=0);
      void* get(int& index,int& size);
      void free(int index, int size);
    
    private:
      int _myTrFd;
      int _nfd = 2;
      pollfd _pfd[2];
      DgramHandler* _handler;
    };
  };
};
#endif

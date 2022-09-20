#ifndef PsAlg_ShMem_ShmemClient_hh
#define PsAlg_ShMem_ShmemClient_hh

#include <poll.h>
#include <stddef.h>
#include <mqueue.h>

namespace XtcData {

    class Dgram;

};

namespace psalg {
  namespace shmem {

    class DgramHandler;

    class ShmemClient {
    public:
      ShmemClient();
      virtual ~ShmemClient();

    public:
      //
      //  tr_index must be unique among clients
      //  unique values of ev_index produce a serial chain of clients sharing events
      //  common values of ev_index produce a set of clients competing for events
      //
      int connect(const char* tag, int tr_index=0);
      void* get(int& index, size_t& size);
      void free(int index, size_t size);

    private:
      void _shutdown();

    private:
      int           _myTrFd;
      int           _nfd = 2;
      pollfd        _pfd[2];
      DgramHandler* _handler;
      unsigned      _numberOfEvQueues;  // number of message queues for events
      mqd_t         _myInputEvQueue;    // message queue for returned events
      mqd_t*        _myOutputEvQueue;   // message queues[nclients] for distributing events
    };
  };
};
#endif

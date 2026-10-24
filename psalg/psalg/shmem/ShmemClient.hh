/**
 * @file
 * @brief Declares psalg::shmem::ShmemClient, the client side of the shared-memory monitor (transitions over TCP, events over message queues).
 */
#ifndef PsAlg_ShMem_ShmemClient_hh
#define PsAlg_ShMem_ShmemClient_hh

#include <poll.h>
#include <stddef.h>
#include <mqueue.h>

/** Namespace of the XTC data-format classes, defined in the xtcdata package; only Dgram is forward-declared here. */
namespace XtcData {

    class Dgram;

};

namespace psalg {
  namespace shmem {

    class DgramHandler;

    /** Client of the shared-memory monitor server: connect() finds the server for a tag, maps its shared memory read-only and opens the transition socket and event queues; get() returns the next datagram and free() hands its buffer back. */
    class ShmemClient {
    public:
      /** Construct unconnected: no socket, queues or handler. */
      ShmemClient();
      /** Close the message queues and the transition socket and delete the handler. */
      virtual ~ShmemClient();

    public:
      //
      //  tr_index must be unique among clients
      //  unique values of ev_index produce a serial chain of clients sharing events
      //  common values of ev_index produce a set of clients competing for events
      //
      /**
       * Drop any previous connection, read the server port from the discovery queue for tag (retrying), connect a TCP socket to 127.0.0.1 on that port (retrying), read the first XtcMonitorMsg, map the shared memory read-only, and open this client's event input and output queues.
       * tr_index is used only in an error message.
       * @return 0 on success; 1 if the socket cannot be created; a positive error count if the first message cannot be read or a queue cannot be opened; -42 if the discovery queue cannot be opened while PSANA_TESTS_SHMEM_TMO is set.
       */
      int connect(const char* tag, int tr_index=0);
      /**
       * Block in poll() until the transition socket or the event queue is readable, serving transitions first, and return a pointer to the datagram in shared memory with index and size set to its buffer index and buffer size.
       * Returns NULL, with index -1 and size 0, on a receive error, a closed socket or an invalid buffer index.
       */
      void* get(int& index, size_t& size);
      /**
       * Same as get(int&, size_t&), but if transitionsOnly is true and poll() returns with no transition pending, set eventSkipped to true and return NULL without reading the event.
       * eventSkipped is never set to false here.
       */
      void* get(int& index, size_t& size, bool& eventSkipped, bool transitionsOnly = false);
      /** Return buffer index (located with size) to the server: for an L1Accept datagram send the message on this client's output event queue, otherwise send it on the transition socket (perror() on failure). */
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

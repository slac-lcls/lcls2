/**
 * @file
 * @brief Declares psalg::shmem::XtcMonitorServer, which serves datagrams to monitoring clients through shared memory.
 */
#ifndef MonReq_XtcMonitorServer_hh
#define MonReq_XtcMonitorServer_hh

//--------------------------------------
//
//  A class for serving datagrams to clients
//  over shared memory.
//
//  The server is configured to setup a fixed number of
//  event queues from which clients may draw events.  Events
//  are pushed to those queues in either of two
//  strategies: serially or distributed.  Serial placement
//  means that events are pushed from the server into the
//  first event queue.  Events that are drawn from that queue
//  and processed by their client will be placed in the next
//  queue for processing by that queue's client.  In this manner,
//  each event is available for processing by all clients.
//  Distributed event placement means that the server will put
//  the first event in the first queue, the second event in
//  the second queue, and so on in a round-robin fashion.  When
//  clients are finished with an event from their queue, the
//  event is returned to the server.  In this mode, events are
//  distributed among clients so that each client sees only a
//  fraction of events.
//
//  The shared memory is divided into equal-size segments where
//  each segment is the target for a datagram(event).  The
//  segments are identified by an index which is distributed
//  to clients via message queues or sockets.  The events are
//  distributed via message queues in the scheme described
//  above.  Shared memory segments for events can be reused
//  once the index returns to the server's message queue.
//
//  Transitions are distributed to clients via a TCP
//  socket with some special consideration for the latency of
//  the client's processing; a transition is passed to a client
//  if that client is no more than one calib cycle behind the
//  current (DAQ) processing state.  A set of transitions necessary
//  to bring a newly connected client up to the current state
//  of the DAQ is cached.  Shared memory segments used for
//  transitions are not reused until all clients have returned
//  the buffer.
//
//-----------------------------------

#include "XtcMonitorMsg.hh"

#include "xtcdata/xtc/TransitionId.hh"

#include <thread>
#include <atomic>
#include <mqueue.h>
#include <queue>
#include <stack>
#include <vector>
#include <poll.h>
#include <time.h>
#include <stddef.h>

namespace XtcData {
  class Dgram;
};

namespace psalg {
  namespace shmem {

    class TransitionCache;

    /**
     * Serves datagrams to clients through shared memory: event buffer indexes go through POSIX message queues (serially or round-robin, see distribute()), transition buffer indexes through TCP sockets, and a TransitionCache keeps the transitions late clients need.
     * The header comment above describes the design. Shared memory, queues and threads are created by the protected _init().
     */
    class XtcMonitorServer {
    public:
      /**
       * Store the sizes, set up the message template (buffer count numberofEvBuffers plus 18 transition buffers), and install handlers for SIGINT, SIGSEGV, SIGABRT, SIGTERM and SIGPIPE that call unlink() and re-raise the signal.
       * Shared memory, queues and threads are not created here (see _init()).
       */
      XtcMonitorServer(const char* tag,
                       unsigned sizeofBuffers,
                       unsigned numberofEvBuffers,
                       unsigned numberofEvQueues);
      /** Stop and join the discovery and task threads, print "Not Unlinking Shared Memory...", call unlink(), and delete the transition cache and internal arrays. */
      virtual ~XtcMonitorServer();
    public:
      /** Return value of events(). */
      enum Result { Handled, /**< Transition: already copied into a transition buffer, or dropped if none was free. */ Deferred  /**< L1Accept: queued for copying by the task thread, or dropped if no event buffer was free; _deleteDatagram() is called when done. */ };
      /**
       * For an L1Accept, take a free event buffer (from the request queue, or stolen from a client queue) and queue dg for copying by the task thread, dropping it if none is free; returns Deferred.
       * For other transitions, get a buffer from the TransitionCache, copy dg into it, on Enable reclaim all event buffers from the client queues, and send the buffer index to every ready client over TCP; returns Handled (also when no buffer is free).
       */
      Result events   (XtcData::Dgram* dg);
      /** Sleep in 1-second steps until the client list is non-empty. */
      void wait       ();
      /**
       * Discovery loop, run on its own thread by _init(): bind a TCP socket on 127.0.0.1 from port 32768 upward, advertise the port on the discovery message queue, and pass each accepted connection to the task thread through a pipe until terminated.
       * Exits the process if the socket cannot be created; aborts on accept errors and on queue errors other than EAGAIN.
       */
      void discover   ();
      /**
       * Task loop, run on its own thread by _init(): set up new clients, move returned event buffers to the request queue, copy queued events into shared memory and send their index to a client queue (serially or round-robin), and handle returned transition buffers and client disconnects until terminated.
       * On exit it shuts down and closes the client sockets.
       */
      void routine    ();
      /** Set the terminate flag, then close all message queues and unlink their names (event, request, shuffle and discovery queues). The shared memory is not unlinked. */
      void unlink     ();
    public:
      /** Choose event placement: true sets the return queue in the message template to the number of event queues (round-robin distribution), false sets it to 0 (serial). Messages already in the request and input queues are rewritten with the new value. */
      void distribute (bool);
    protected:
      int  _init             ();
    private:
      void _initialize_client();
      mqd_t _openQueue       (const char* name, mq_attr&);
      void _flushQueue       (mqd_t q);
      void _flushQueue       (mqd_t q, char* m, unsigned sz);
      void _moveQueue        (mqd_t iq, mqd_t oq);
      void _replQueue        (mqd_t q, unsigned rq);
      bool _send             (XtcData::Dgram*);
      void _update           (int, XtcData::TransitionId::Value);
      void _clearDest        (mqd_t);
    private:
      virtual void _copyDatagram   (XtcData::Dgram* dg, char*, size_t);
      virtual void _deleteDatagram (XtcData::Dgram* dg);
      virtual void _requestDatagram();
    private:
      const char*       _tag;               // name of the complete shared memory segment
      size_t            _sizeOfBuffers;     // size of each shared memory datagram buffer
      unsigned          _numberOfEvBuffers; // number of shared memory buffers for events
      unsigned          _numberOfEvQueues;  // number of message queues for events
      char*             _myShm;             // the pointer to start of shared memory
      XtcMonitorMsg     _myMsg;             // template for messages
      mqd_t             _discoveryQueue;    // message queue for clients to get
                                            // the TCP port for initiating connections
      mqd_t             _myInputEvQueue;    // message queue for returned events
      mqd_t*            _myOutputEvQueue;   // message queues[nclients] for distributing events
      std::vector<int>  _myTrFd;            // TCP sockets to clients for distributing
                                            // transitions and detecting disconnects.
      std::vector<int>  _msgDest;           // last client to which the buffer was sent
      TransitionCache*  _transitionCache;
      int               _initFd;
      pollfd*           _pfd;               /* poll descriptors for:
                                            **   0  new client connections
                                            **   1  buffer returned from client
                                            **   2  events to be distributed
                                            **   3+ transition send/receive  */
      int               _nfd;
      mqd_t             _shuffleQueue;      // message queue for pre-distribution event processing
      mqd_t             _requestQueue;      // message queue for buffers awaiting request completion
      std::atomic<bool> _terminate;         // Flag for causing subthreads to exit
      std::thread       _discThread;        // thread for receiving new client connections
      std::thread       _taskThread;        // thread for datagram distribution
      unsigned          _ievt;              // event vector
    };
  };
};

#endif

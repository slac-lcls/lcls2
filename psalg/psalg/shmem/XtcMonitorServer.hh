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

namespace XtcData {
  class Dgram;
};

namespace psalg {
  namespace shmem {

    class TransitionCache;

    class XtcMonitorServer {
    public:
      XtcMonitorServer(const char* tag,
                       unsigned sizeofBuffers,
                       unsigned numberofEvBuffers,
                       unsigned numberofEvQueues);
      virtual ~XtcMonitorServer();
    public:
      enum Result { Handled, Deferred };
      Result events   (XtcData::Dgram* dg);
      void wait       ();
      void discover   ();
      void routine    ();
      void unlink     ();
    public:
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
      virtual void _copyDatagram  (XtcData::Dgram* dg, char*);
      virtual void _deleteDatagram(XtcData::Dgram* dg, int idx);
      virtual void _deleteDatagram(XtcData::Dgram* dg);
      virtual void _requestDatagram(int idx);
      virtual void _requestDatagram();
    private:
      const char*       _tag;               // name of the complete shared memory segment
      unsigned          _sizeOfBuffers;     // size of each shared memory datagram buffer
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
      timespec          _tmo;
      std::atomic<bool> _terminate;         // Flag for causing subthreads to exit
      std::thread       _discThread;        // thread for receiving new client connections
      std::thread       _taskThread;        // thread for datagram distribution
      unsigned          _ievt;              // event vector
    };
  };
};

#endif

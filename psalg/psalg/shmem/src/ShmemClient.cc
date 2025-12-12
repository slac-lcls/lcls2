#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#ifdef _POSIX_MESSAGE_PASSING
#include <mqueue.h>
#endif

#include <sys/socket.h>
#include <arpa/inet.h>

#include "xtcdata/xtc/Dgram.hh"
#include "ShmemClient.hh"
#include "XtcMonitorMsg.hh"

#include <poll.h>
#include <time.h>

//#define DBUG
//#define DBUG2

enum {PERMS = S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH};
enum {PERMS_IN  = S_IRUSR|S_IRGRP|S_IROTH};
enum {PERMS_OUT  = S_IWUSR|S_IWGRP|S_IWOTH};
enum {OFLAGS = O_RDONLY};

/*
** ++
**
**
** --
*/

static mqd_t _openQueue(const char* name, unsigned flags, unsigned perms,
                        bool lwait=true)
{
  struct mq_attr mymq_attr;
  mqd_t queue;
  bool first = true;
  const char* test_tmo_env = std::getenv("PSANA_TESTS_SHMEM_TMO");
  int tmo = 0;
  if (test_tmo_env) {
    tmo = std::atoi(test_tmo_env);
  }
  auto start = std::chrono::steady_clock::now();
  while(1) {
    queue = mq_open(name, flags, perms, &mymq_attr);
    if (queue == (mqd_t)-1) {
      if (!lwait) {
        printf("Failed to open queue %s\n",name);
        break;
      }
      if (first) {
        first = false;
        printf("Waiting for queue %s to appear\n",name);
      }
      if (tmo) {
        auto now = std::chrono::steady_clock::now();
        if (now - start >= std::chrono::seconds(tmo)) {
          printf("Failed to open queue %s. Breaking after PSANA_TESTS_SHMEM_TMO=%d sec\n",name,tmo);
          break;
        }
      }
      sleep(1);
    }
    else {
      printf("Opened queue %s (%d)\n",name,queue);
      break;
    }
  }
  return queue;
}

/*
** ++
**
**
** --
*/

namespace psalg {
  namespace shmem {
    class DgramHandler {

    private:
      ShmemClient&            _client;
      XtcMonitorMsg                _myMsg;
      int                          _trfd;
      mqd_t                        _evqin;
      mqd_t*                       _evqout;
      unsigned                     _ev_index;
      const char*                  _tag;
      char*                        _shm;
      timespec                     _tmo;

    public:
      ~DgramHandler() {}

      /*
      ** ++
      **
      **
      ** --
      */

      DgramHandler(ShmemClient& client, XtcMonitorMsg myMsg,
                   int trfd, mqd_t evqin, mqd_t* evqout, unsigned ev_index,
                   const char* tag, char* myShm) :
        _client(client), _myMsg(myMsg),
        _trfd(trfd), _evqin(evqin), _evqout(evqout), _ev_index(ev_index),
        _tag(tag), _shm(myShm)
      {
        _tmo.tv_sec = _tmo.tv_nsec = 0;
      }


      /*
      ** ++
      **
      **
      ** --
      */

      void free(int index, size_t size) {
          XtcMonitorMsg myMsg = _myMsg;
          myMsg.bufferIndex(index);
          mqd_t* oq = _evqout;
          unsigned ioq = _ev_index;
          unsigned priority(0);
          XtcData::Dgram* dg = (XtcData::Dgram*) (_shm + (size * (size_t)index));

          if(dg->service()==XtcData::TransitionId::L1Accept) {
#ifdef DBUG
              printf("ShmemClient DgramHandler free dgram index %d size %zd\n",index,size);
#endif
              mq_timedsend(oq[ioq], (const char *)&myMsg, sizeof(myMsg), priority, &_tmo);
          } else {
              if(::send(_trfd,(char*)&myMsg,sizeof(myMsg),MSG_NOSIGNAL)<0) {
                  // cpo: we can get an error if the server exits
                  // not clear how we should handle this.  keep
                  // the server alive? print a warning?
                  perror("ShmemClient.cc: transition send error (can happen if server exits)");
              }
          }
      }


      /*
      ** ++
      **
      **
      ** --
      */

      XtcData::Dgram* transition(int &index, size_t &size) {
        XtcMonitorMsg myMsg;
        int nb = ::recv(_trfd, (char*)&myMsg, sizeof(myMsg), 0);
        if (nb < 0) {
          perror("transition receive");
          return NULL;
        }
        if (nb == 0) {
#ifdef DBUG
          printf("Received tr disconnect [%d]\n",_trfd);
#endif
          return NULL;
        }
        int i = myMsg.bufferIndex();
#ifdef DBUG
        printf("Received tr buffer %d [%d]\n",i,_trfd);
#endif
        if ( (i>=0) && (i<myMsg.numberOfBuffers())) {
          XtcData::Dgram* dg = (XtcData::Dgram*) (_shm + (myMsg.sizeOfBuffers() * (size_t)i));
#ifdef DBUG2
          printf("*** received transition id %d (%s)\n",dg->service(), XtcData::TransitionId::name(dg->service()));
#endif
          index = i;
          size = myMsg.sizeOfBuffers();

          return dg;
        }
        else {
          fprintf(stderr, "ILLEGAL TR BUFFER INDEX %d numBuffers %d\n", i,myMsg.numberOfBuffers());
          uint32_t* p = reinterpret_cast<uint32_t*>(&myMsg);
          fprintf(stderr, "XtcMonitorMsg: %x/%x/%x/%x [%d]\n",p[0],p[1],p[2],p[3],nb);
          return NULL;
        }
        return NULL;
      }


      /*
      ** ++
      **
      **
      ** --
      */

      XtcData::Dgram* event(int &index, size_t &size) {
        mqd_t  iq = _evqin;

        XtcMonitorMsg myMsg;
        unsigned priority(0);
        int nb;
        if ((nb=mq_receive(iq, (char*)&myMsg, sizeof(myMsg), &priority)) < 0) {
          perror("mq_receive buffer");
          return NULL;
        }
        else {
          int i = myMsg.bufferIndex();
#ifdef DBUG
          printf("*** Received ev buffer %d [%d] numBuffers %d size %zd\n",i,iq,myMsg.numberOfBuffers(),myMsg.sizeOfBuffers());
#endif
          if ( (i>=0) && (i<myMsg.numberOfBuffers())) {
            XtcData::Dgram* dg = (XtcData::Dgram*) (_shm + (myMsg.sizeOfBuffers() * (size_t)i));
#ifdef DBUG2
            static uint64_t nevt = 0;
            printf("*** received ev svc %d (%s) %u.%09u, nL1A %lu\n",dg->service(), XtcData::TransitionId::name(dg->service()), dg->time.seconds(), dg->time.nanoseconds(), nevt);
            if (dg->service() == XtcData::TransitionId::L1Accept)  ++nevt;
#endif
            index = i;
            size = myMsg.sizeOfBuffers();
            return dg;
          }
          else {
            fprintf(stderr, "ILLEGAL EV BUFFER INDEX %d numBuffers %d\n", i,myMsg.numberOfBuffers());
            uint32_t* p = reinterpret_cast<uint32_t*>(&myMsg);
            fprintf(stderr, "XtcMonitorMsg: %x/%x/%x/%x [%d]\n",p[0],p[1],p[2],p[3],nb);
            return NULL;
          }
        }
        return NULL;
      }
    };
  };
};

/*
** ++
**
**
** --
*/

using namespace XtcData;
using namespace psalg::shmem;

ShmemClient::ShmemClient() :
  _myTrFd           (-1),
  _handler          (0),
  _numberOfEvQueues (0),
  _myInputEvQueue   ((mqd_t)-1),
  _myOutputEvQueue  (nullptr)
{
}

ShmemClient::~ShmemClient()
{
  _shutdown();
}

void ShmemClient::_shutdown()
{
  if (_myInputEvQueue != (mqd_t)-1)        mq_close(_myInputEvQueue);

  for (unsigned i = 0; i < _numberOfEvQueues; ++i)
  {
    if (_myOutputEvQueue[i] != (mqd_t)-1)  mq_close(_myOutputEvQueue[i]);
  }

  // Avoid race with server and let it unlink the mqueues

  if (_handler)          { delete    _handler;          _handler = 0; }
  if (_myOutputEvQueue)  { delete [] _myOutputEvQueue;  _myOutputEvQueue = nullptr; }

  if (!(_myTrFd < 0))    { ::close(_myTrFd);            _myTrFd = -1; }
}

/*
** ++
**
**
** --
*/

void ShmemClient::free(int index, size_t size)
{
  _handler->free(index,size);
}

/*
* Return a new transition or l1accept datagram
* @param index Buffer index updated by reference. Used by client app to free
*        (release) buffer once done with it.
* @param size Buffer size updated by reference.
* @param eventSkipped If transitionsOnly below is true, this reference will be set
*        to true if an event (l1Accept) was actually skipped. This is to allow client
*        applications to determine if NULL is an expected return value.
* @param transitionsOnly If true, only return transitions (if available), skipping
*        any l1accepts that may be pending. Can be used by client applications to
*        ensure reading of all transitions without also draining off any l1accepts
*        that may otherwise be discarded. Default: false (see header)
*
*/
void* ShmemClient::get(int& index, size_t& size, bool& eventSkipped, bool transitionsOnly)
{
  index = -1;
  size = 0;

  while (1) {
    if (::poll(_pfd, _nfd, -1) > 0) {
          if (_pfd[0].revents & POLLIN) { // Transition
            return _handler->transition(index, size);
          } else if (transitionsOnly) {
            // Application requested to skip any events (L1Accept)
            // Instead of blocking return NULL, application must handle this
            eventSkipped = true;
            return NULL;
          } else if (_pfd[1].revents & POLLIN) { // Event
            return _handler->event(index, size);
          }
    }
  }
  return NULL;
}

void* ShmemClient::get(int& index, size_t& size)
{
  index = -1;
  size = 0;

  while (1) {
    if (::poll(_pfd, _nfd, -1) > 0) {
      if (_pfd[0].revents & POLLIN) { // Transition
        return _handler->transition(index,size);
      }
      else if (_pfd[1].revents & POLLIN) { // Event
        return _handler->event(index,size);
      }
    }
  }
  return NULL;
}

/*
** ++
**
**
** --
*/

int ShmemClient::connect(const char* tag, int tr_index) {
  int error = 0;
  char* qname = new char[128];

  umask(0);   // Need this to set group/other write permissions on mqueue

  // In case resources are still lingering from a previous call, shut down first
  _shutdown();

  XtcMonitorMsg myMsg;
  unsigned priority;

  //
  //  Request initialization
  //

  _myTrFd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (_myTrFd < 0) {
    perror("Opening myTrFd socket");
    delete[] qname;
    return 1;
  }

  const char* test_tmo_env = std::getenv("PSANA_TESTS_SHMEM_TMO");
  int test_tmo = 0;
  if (test_tmo_env) {
    test_tmo = std::atoi(test_tmo_env);
  }

  XtcMonitorMsg::discoveryQueue(tag,qname);
  ssize_t rv = -1;
  do {
      mqd_t discoveryQueue = _openQueue(qname, O_RDONLY, PERMS_IN);
      if (discoveryQueue == (mqd_t)-1) {
          sleep(1);
          if (test_tmo) {
              printf("Exiting early due to requested timeout.\n");
              _shutdown();
              return -42;
          } else {
              continue;
          }
      }

      printf("[%p] Reading discoveryQueue %d\n",this, discoveryQueue);
      timespec tmo;
      clock_gettime(CLOCK_REALTIME,&tmo);
      tmo.tv_sec++;
      rv = mq_timedreceive(discoveryQueue, (char*)&myMsg, sizeof(myMsg), &priority, &tmo);
      mq_close(discoveryQueue);
  } while (rv<0);

  unsigned port = myMsg.bufferIndex();

  sockaddr_in saddr;
  saddr.sin_family = AF_INET;
  saddr.sin_addr.s_addr = htonl(0x7f000001);
  saddr.sin_port        = htons(port);

  printf("[%p] Connecting to XtcMonitor server on port %d (%d)\n",this,port,_myTrFd);
  unsigned retries = 0;
  while (::connect(_myTrFd, (sockaddr*)&saddr, sizeof(saddr)) < 0) {
      if (!retries++)  perror("Error connecting myTrFd socket (will retry)");
      sleep(1);
  }
  printf("[%p] Connected to XtcMonitor server on port %d (%d) after %d retries\n",
         this,port,_myTrFd,retries);

#ifdef DBUG
  socklen_t addrlen = sizeof(sockaddr_in);
  sockaddr_in name;
  ::getsockname(_myTrFd, (sockaddr*)&name, &addrlen);
  printf("[%p] Connected to %08x.%d [%d] from %08x.%d\n", this,
         ntohl(saddr.sin_addr.s_addr),ntohs(saddr.sin_port),_myTrFd,
         ntohl(name.sin_addr.s_addr),ntohs(name.sin_port));
#endif

  ssize_t rc;
  if ((rc=::read(_myTrFd,&myMsg,sizeof(myMsg)))!=sizeof(myMsg)) {
    if (rc < 0)  perror("read failed");
    else         printf("read returned %zd of %zd bytes\n", rc, sizeof(myMsg));
    delete[] qname;
    return ++error;
  }

  //
  //  Initialize shared memory from first message
  //
  size_t sizeOfShm = myMsg.numberOfBuffers() * myMsg.sizeOfBuffers();
  size_t pageSize  = (unsigned)sysconf(_SC_PAGESIZE);
  size_t remainder = sizeOfShm % pageSize;
  if (remainder)
    sizeOfShm += pageSize - remainder;

  XtcMonitorMsg::sharedMemoryName(tag, qname);
  printf("Opening shared memory %s of size 0x%zx (0x%x * 0x%zx)\n",
         qname,sizeOfShm,myMsg.numberOfBuffers(),myMsg.sizeOfBuffers());

  int shm = shm_open(qname, OFLAGS, PERMS_IN);
  if (shm < 0) perror("shm_open");
  char* myShm = (char*)mmap(NULL, sizeOfShm, PROT_READ, MAP_SHARED, shm, 0);
  if (myShm == MAP_FAILED) perror("mmap");
  else printf("Shared memory at %p\n", (void*)myShm);

  close(shm);  // Done with the file descriptor

  int ev_index = myMsg.bufferIndex();
  XtcMonitorMsg::eventInputQueue(tag,ev_index,qname);
  _myInputEvQueue = _openQueue(qname, O_RDONLY, PERMS_IN);
  if (_myInputEvQueue == (mqd_t)-1)
    error++;

  _numberOfEvQueues = myMsg.numberOfQueues()+1;
  _myOutputEvQueue = new mqd_t[_numberOfEvQueues];
  for(int i=0; i<=myMsg.numberOfQueues(); i++)
    _myOutputEvQueue[i]=-1;

  if (myMsg.serial()) {
    XtcMonitorMsg::eventOutputQueue(tag,ev_index,qname);
    _myOutputEvQueue[ev_index] = _openQueue(qname, O_WRONLY, PERMS_OUT);
    if (_myOutputEvQueue[ev_index] == (mqd_t)-1)
      error++;
  }
  else {
    XtcMonitorMsg::eventInputQueue(tag,myMsg.return_queue(),qname);
    _myOutputEvQueue[ev_index] = _openQueue(qname, O_WRONLY, PERMS_OUT);
    if (_myOutputEvQueue[ev_index] == (mqd_t)-1)
      error++;
  }
  delete[] qname;

  if (error) {
    fprintf(stderr, "Could not open at least one message queue!\n");
    fprintf(stderr, "tag %s, tr_index %d, ev_index %d\n",tag,tr_index,ev_index);
    return error;
  }

  //
  //  Handle all transitions first, then events
  //
  _pfd[0].fd      = _myTrFd;
  _pfd[0].events  = POLLIN | POLLERR;
  _pfd[0].revents = 0;
  _pfd[1].fd      = _myInputEvQueue;
  _pfd[1].events  = POLLIN | POLLERR;
  _pfd[1].revents = 0;
  _nfd = 2;

  // Assumption: myMsg numberOfBuffers and sizeOfBuffers are constant
  // throughout lifetime of client connection.
  _handler = new DgramHandler(*this,
                              myMsg,
                              _myTrFd,
                              _myInputEvQueue, _myOutputEvQueue, ev_index,
                              tag,myShm);

  return 0;
}

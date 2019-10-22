#include "XtcMonitorServer.hh"
#include "TransitionCache.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <unistd.h>
#include <semaphore.h>
#ifdef _POSIX_MESSAGE_PASSING
#include <mqueue.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <time.h>

#include <sys/socket.h>
#include <arpa/inet.h>

#include <list>

//#define DBUG
//#define NO_STEAL

using std::queue;
using std::stack;

static const unsigned numberofTrBuffers=18;

//
//  Recover any shared memory buffer older than 10 seconds
//
static const unsigned TMO_SEC = 10;

#define PERMS (S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH)
#define PERMS_IN (S_IRUSR|S_IRGRP|S_IROTH)
#define OFLAGS (O_CREAT|O_RDWR)

namespace psalg {
  namespace shmem {
    class ShMsg {
    public:
      ShMsg() {}
      ShMsg(const XtcMonitorMsg&  m,
            XtcData::Dgram* dg) : _m(m), _dg(dg) {}
      ~ShMsg() {}
    public:
      const XtcMonitorMsg&  msg() const { return _m; }
      XtcData::Dgram* dg () const { return _dg; }
    private:
      XtcMonitorMsg   _m;
      XtcData::Dgram* _dg;
    };
  };
};

using namespace XtcData;
using namespace psalg::shmem;

static XtcMonitorServer* apps;

static struct sigaction old_actions[64];

void sigfunc(int sig_no) {
  static bool _handled=false;
  if (!_handled) {
    _handled = true;
    printf("handling signal %d app %p\n",sig_no,apps);
    if (apps) {
      apps->unlink();
      apps = 0;
    }
    else
      printf("nothing to do\n");

    printf("done with signal %d\n",sig_no);

    sigaction(sig_no,&old_actions[sig_no],NULL);
    raise(sig_no);
  }
}


XtcMonitorServer::XtcMonitorServer(const char* tag,
                                   unsigned sizeofBuffers,
                                   unsigned numberofEvBuffers,
                                   unsigned numberofEvQueues) :
  _tag              (tag),
  _sizeOfBuffers    (sizeofBuffers),
  _numberOfEvBuffers(numberofEvBuffers),
  _numberOfEvQueues (numberofEvQueues),
  _myShm            (0),
  _discoveryQueue   (-1),
  _myInputEvQueue   (-1),
  _myOutputEvQueue  (new mqd_t[numberofEvQueues]),
  _myTrFd           (0),
  _msgDest          (numberofEvBuffers),
  _pfd              (new pollfd[32]),
  _nfd              (3),
  _shuffleQueue     (-1),
  _requestQueue     (-1),
  _ievt             (0)
{
  _myMsg.numberOfBuffers(numberofEvBuffers+numberofTrBuffers);
  _myMsg.numberOfQueues (numberofEvQueues);
  _myMsg.sizeOfBuffers  (sizeofBuffers);
  _myMsg.return_queue   (0);

  _tmo.tv_sec  = 0;
  _tmo.tv_nsec = 0;

  apps = this;

  struct sigaction int_action;

  int_action.sa_handler = sigfunc;
  sigemptyset(&int_action.sa_mask);
  int_action.sa_flags = 0;
  int_action.sa_flags |= SA_RESTART;

#define REGISTER(t) {                                   \
    if (sigaction(t, &int_action, &old_actions[t]) > 0) \
      printf("Couldn't set up #t handler\n");           \
  }

  REGISTER(SIGINT);
  REGISTER(SIGSEGV);
  REGISTER(SIGABRT);
  REGISTER(SIGTERM);
  REGISTER(SIGPIPE);

#undef REGISTER
}

XtcMonitorServer::~XtcMonitorServer()
{
  _terminate.store(true, std::memory_order_release);
  if (_discThread.joinable())  _discThread.join();
  if (_taskThread.joinable())  _taskThread.join();

  printf("Not Unlinking Shared Memory... \n");

  unlink();
  apps = 0;

  delete    _transitionCache;
  delete [] _myOutputEvQueue;
  delete [] _pfd;
}

void XtcMonitorServer::distribute(bool l)
{
  _myMsg.return_queue( l ? _numberOfEvQueues : 0 );

  _flushQueue(_myInputEvQueue);
  _flushQueue(_requestQueue);
  for(unsigned i=0; i<_numberOfEvBuffers; i++) {
    _myMsg.bufferIndex(i);
    if (mq_timedsend(_myInputEvQueue, (const char*)&_myMsg, sizeof(_myMsg), 0, &_tmo)<0)
      perror("XtcMonitorServer distribute failed to queue buffers to input");
  }
}

bool XtcMonitorServer::_send(Dgram* dg)
{
  //
  //  For reasons I don't yet understand, sometimes the message queues
  //  are opened in blocking mode.  So, I use mq_timedreceive
  //  with a 0 timeout to avoid blocking.
  //
  XtcMonitorMsg msg;
  const timespec no_wait={0,0};
  int r = mq_timedreceive(_requestQueue, (char*)&msg, sizeof(msg), NULL,
                          &no_wait);

#ifndef NO_STEAL
  static unsigned _nsteals=0;
  if (r>0)
    ;
  else {
    if (r<0) ; // perror("Error reading input event queue");
    for(unsigned i=0; i<_numberOfEvQueues; i++) {
      unsigned iq = (i+_nsteals)%_numberOfEvQueues; // fairness
      r=mq_timedreceive(_myOutputEvQueue[iq], (char*)&msg, sizeof(msg), NULL,
                        &no_wait);
      if (r>0) break;
      if (r<0) ; // perror("Error reading output event queue");
    }
    _nsteals++;
  }
#endif

  if (r>0) {
    _msgDest[msg.bufferIndex()]=-1;
    ShMsg m(msg, dg);
    if (mq_timedsend(_shuffleQueue, (const char*)&m, sizeof(m), 0, &_tmo)) {
      printf("ShuffleQ timed out\n");
      _deleteDatagram(dg, msg.bufferIndex());
    }
  }
  else {
    // No shared memory buffer found.  Dropping new event.
    _deleteDatagram(dg, -1);
  }
  return true;
}

XtcMonitorServer::Result XtcMonitorServer::events(Dgram* dg)
{
  Dgram& dgrm = *dg;
  // Revisit: Moves to _copyDatagram() since dg describes the Dgram but isn't it
  //if (sizeof(dgrm)+dgrm.xtc.sizeofPayload() > _sizeOfBuffers) {
  //  printf("XtcMonitorServer skipping %s with payload size %d - too large\n",
  //         TransitionId::name(dgrm.seq.service()), dgrm.xtc.sizeofPayload());
  //  //    return Handled;
  //  exit(1);
  //}

  TransitionId::Value trid = dgrm.seq.service();

  if (trid == TransitionId::L1Accept) {
    _send(dg);
    return Deferred;
  }
  else {

    int itr = _transitionCache->allocate(trid);

    if (itr < 0) {
      printf("No buffers available for transition!\n");
      _transitionCache->dump();
      abort();
    }

    int ibuffer = itr + _numberOfEvBuffers;

    _myMsg.bufferIndex(ibuffer);
    _copyDatagram(dg, _myShm + _sizeOfBuffers*ibuffer);

    if (trid == TransitionId::Enable) {
      //
      //  Steal all event buffers from the clients
      //
      for(unsigned i=0; i<_numberOfEvQueues; i++)
        _moveQueue(_myOutputEvQueue[i], _myInputEvQueue);
    }

    //
    //  Broadcast the transition to all ready clients
    //
    for(unsigned i=0; i<_myTrFd.size(); i++) {
      int oq = _myTrFd[i];
      if (oq == -1 || !_transitionCache->allocate(itr,i))
        continue;
      if (::send(oq, (const char*)&_myMsg, sizeof(_myMsg), 0) < 0) {
        perror("Error sending transition");
        _transitionCache->deallocate(itr,i);
      }
    }

  }
  return Handled;
}

void XtcMonitorServer::wait()
{
  //
  //  Wait for clients
  //
  while(! _myTrFd.size())
    sleep(1);

}

void XtcMonitorServer::discover()
{
  int fd;
  if ( (fd = ::socket(AF_INET, SOCK_STREAM, 0))<0 ) {
    perror("Create discovery socket failed");
    exit(1);
  }

  pollfd pfd;
  int    nfd = 1;
  pfd.fd      = fd;
  pfd.events  = POLLIN;
  pfd.revents = 0;

  //  assign an ephemeral port
  unsigned port = 32768;
  sockaddr_in saddr;
  saddr.sin_family      = AF_INET;
  saddr.sin_addr.s_addr = htonl(0x7f000001);
  while(1) {
    saddr.sin_port        = htons(port);
#ifdef DBUG
    printf("Trying to bind to port %d\n",port);
#endif
    if (::bind(fd,(sockaddr*)&saddr,sizeof(saddr))>=0)
      break;
#ifdef DBUG
    else
      perror("bind failed");
#endif
    port++;
  }
  printf("Awaiting XtcMonitor clients on port %d (%d)\n",port,fd);

  const char* p = _tag;
  char* fromQname  = new char[128];
  XtcMonitorMsg::discoveryQueue(p, fromQname);
  struct mq_attr q_attr;
  q_attr.mq_maxmsg  = 2;
  q_attr.mq_msgsize = (long int)sizeof(XtcMonitorMsg);
  q_attr.mq_flags   = O_NONBLOCK;
  _discoveryQueue  = _openQueue(fromQname,q_attr);
  delete[] fromQname;
  _flushQueue(_discoveryQueue);

  if (::listen(fd,10)<0)
    printf("ConnectionManager listen failed\n");
  else {
    while(!_terminate.load(std::memory_order_relaxed)) {
      timespec tv; tv.tv_sec=tv.tv_nsec=0;
      XtcMonitorMsg m(port);
      if (mq_timedsend(_discoveryQueue,(const char*)&m,sizeof(m),0,&_tmo)<0) {
        perror("Error advertising discovery port");
        abort();
      }

      // Wait for a connection
      while(!_terminate.load(std::memory_order_relaxed)) {
        int rc;
        int tmo = 1000;
        if ((rc = ::poll(&pfd,nfd,tmo)) > 0) {
          if (pfd.revents & POLLIN) {
            sockaddr_in client;
            socklen_t   len = sizeof(sockaddr_in);
            int s = ::accept(fd,(sockaddr*)&client,&len);
            if (s<0) {
              perror("XtcMonitorServer discovery accept failed");
              abort();
            }

#ifdef DBUG
            printf("Accepted connection from %x.%d on socket %d\n",
                   ntohl(client.sin_addr.s_addr),ntohs(client.sin_port),s);
            printf("Writing to %d\n",_initFd);
#endif

            //  Post connection request to taskThread
            ::write(_initFd,&s,sizeof(s));

            // Advertise the discovery port again
            break;
          }
        }
      }
    }
  }
  ::close(fd);
  ::close(_initFd);
}

void XtcMonitorServer::routine()
{
  while(!_terminate.load(std::memory_order_relaxed)) {

    int tmo = 1000;                     // ms
    if ((::poll(_pfd,_nfd,tmo)) > 0) {

      if (_pfd[0].revents & POLLIN)
        _initialize_client();

      //
      //  Handle buffers returned from client
      //
      if (_pfd[1].revents & POLLIN) {
        XtcMonitorMsg msg;
        const timespec no_wait={0,0};
        while(mq_timedreceive(_myInputEvQueue, (char*)&msg, sizeof(msg), NULL, &no_wait) > 0) {
          if (mq_timedsend(_requestQueue, (const char*)&msg, sizeof(msg), 0, &_tmo))
            perror("Writing to requestQ");
          else
            _requestDatagram(msg.bufferIndex());
        }
      }

      //
      //  Handle events ready for distribution
      //
      if (_pfd[2].revents & POLLIN) {
        ShMsg m;
        if (mq_receive(_shuffleQueue, (char*)&m, sizeof(m), NULL) < 0)
          perror("mq_receive");

        _copyDatagram(m.dg(),_myShm+_sizeOfBuffers*m.msg().bufferIndex());
        _deleteDatagram(m.dg(), m.msg().bufferIndex());

        if (m.msg().serial()) {
          //
          //  Send this event to the first available client
          //
          for(unsigned i=0; i<=_numberOfEvQueues; i++)
            if (mq_timedsend(_myOutputEvQueue[i], (const char*)&m.msg(), sizeof(m.msg()), 0, &_tmo))
              ; //          printf("outputEv timed out to client %d\n",i);
            else {
              _msgDest[m.msg().bufferIndex()]=i;
              break;
            }
        }
        else {
          //
          //  Send this event to the next client around (round-robin)
          //
          bool lsent=false;
          for(unsigned i=0; i<_numberOfEvQueues; i++) {
            int oc = _ievt++%_numberOfEvQueues;
            int oq = _myOutputEvQueue[oc];
            if (mq_timedsend(oq, (const char*)&m.msg(), sizeof(m.msg()), 0, &_tmo))
              ;
            else {
              _msgDest[m.msg().bufferIndex()]=oc;
              lsent=true;
              break;
            }
          }
          if (!lsent) {
            if (mq_timedsend(_myInputEvQueue, (const char*)&m.msg(), sizeof(m.msg()), 0, &_tmo))
              perror("Unable to distribute or reclaim event");
          }
        }
      }

      //
      //  Receive transitions back from clients
      //  or handle client disconnects
      //
      for(int i=3; i<_nfd; i++) {
        if (_pfd[i].revents & POLLIN) {
          int r;
          XtcMonitorMsg msg;
          while((r=::recv(_pfd[i].fd, (char*)&msg, sizeof(msg), MSG_DONTWAIT))>=0) {
            for(unsigned q=0; q<_myTrFd.size(); q++)
              if (_myTrFd[q]==_pfd[i].fd) {
                if (r > 0) {
                  int itr=msg.bufferIndex()-_numberOfEvBuffers;
                  if (_transitionCache->deallocate(itr,q))
                    _update(q,reinterpret_cast<Dgram*>(_myShm+_sizeOfBuffers*msg.bufferIndex())->seq.service());
                }
                else { // retire client
                  printf("Retiring client %d [%d]\n",q,_pfd[i].fd);
                  //  Recover buffers last sent to this client

                  //  First, account for the ones waiting in our input queue
                  _clearDest(_myInputEvQueue);
                  _clearDest(_requestQueue);

                  //  Recover the buffers still queued to the retired client
                  _moveQueue(_myOutputEvQueue[q], _myInputEvQueue);

                  //  Force recovery of those still outstanding to the retired client
                  for(int j=0; j<int(_msgDest.size()); j++)
                    if (_msgDest[j]==int(q)) {
                      printf("Recovering buffer %d\n",j);
                      msg = _myMsg;
                      msg.bufferIndex(j);
                      if (mq_timedsend(_myInputEvQueue, (const char*)&msg, sizeof(msg), 0, &_tmo)<0)
                        perror("Failed to recover buffer queued to retired client");
                      else
                        _msgDest[j]=-1;
                    }

                  _myTrFd[q]=-1;
                  //  Clear the transition tracking for this client
                  _transitionCache->deallocate(q);
                  _nfd--;
                  for(int j=i; j<_nfd; j++)
                    _pfd[j] = _pfd[j+1];
                  i--;
                }
                break;
              }
          }
        }
      }
    }
  }

  for(auto i = 3; i < _nfd; ++i) {
    ::shutdown(_pfd[i].fd, SHUT_RDWR); // Induce the client to exit
    ::close(_pfd[i].fd);
  }
  ::close(_pfd[0].fd);
}

void XtcMonitorServer::_clearDest(mqd_t queue)
{
  XtcMonitorMsg msg;
  const timespec no_wait={0,0};
  std::list<int> indices;
  while(mq_timedreceive(queue, (char*)&msg,
                        sizeof(msg), NULL, &no_wait)>0)
    indices.push_back(msg.bufferIndex());

  for(std::list<int>::iterator it=indices.begin();
      it!=indices.end(); it++) {
    _msgDest[*it]=-1;
    msg.bufferIndex(*it);
    if (mq_timedsend(queue, (char*)&msg,sizeof(msg),0,&no_wait)<0) {
      perror("Accounting input queue buffers");
      printf("May have lost buffer %d\n",*it);
    }
  }
}


int XtcMonitorServer::_init()
{
  const char* p = _tag;
  char* shmName    = new char[128];
  char* toQname    = new char[128];
  char* fromQname  = new char[128];

  sprintf(shmName  , "/PdsMonitorSharedMemory_%s",p);
  unsigned pageSize = (unsigned)sysconf(_SC_PAGESIZE);

  int ret = 0;
  size_t sizeOfShm = size_t(_numberOfEvBuffers + numberofTrBuffers) * _sizeOfBuffers;
  unsigned remainder = sizeOfShm%pageSize;
  if (remainder) sizeOfShm += pageSize - remainder;

  umask(1);  // try to enable world members to open these devices.

  int shm = shm_open(shmName, OFLAGS, PERMS);
  if (shm < 0) {ret++; perror("shm_open");}

  if ((ftruncate(shm, sizeOfShm))<0) {ret++; perror("ftruncate");}

  _myShm = (char*)mmap(NULL, sizeOfShm, PROT_READ|PROT_WRITE, MAP_SHARED, shm, 0);
  if (_myShm == MAP_FAILED) {ret++; perror("mmap");}

  _transitionCache = new TransitionCache(_myShm+_numberOfEvBuffers*_sizeOfBuffers,
                                         _sizeOfBuffers,
                                         numberofTrBuffers);

  mq_attr q_attr;

  q_attr.mq_maxmsg  = _numberOfEvBuffers;
  q_attr.mq_msgsize = (long int)sizeof(XtcMonitorMsg);
  q_attr.mq_flags   = O_NONBLOCK;

  XtcMonitorMsg::eventOutputQueue(p,_numberOfEvQueues-1,toQname);
  _flushQueue(_myInputEvQueue  = _openQueue(toQname,q_attr));

  for(unsigned i=0; i<_numberOfEvBuffers; i++) {
    _myMsg.bufferIndex(i);
    _msgDest[i]=-1;
    if (mq_timedsend(_myInputEvQueue, (const char*)&_myMsg, sizeof(_myMsg), 0, &_tmo)<0)
      perror("Failed to queue buffer to input queue (initialize)");
  }

  q_attr.mq_maxmsg  = _numberOfEvBuffers / _numberOfEvQueues;
  q_attr.mq_msgsize = (long int)sizeof(XtcMonitorMsg);
  q_attr.mq_flags   = O_NONBLOCK;

  for(unsigned i=0; i<_numberOfEvQueues; i++) {
    XtcMonitorMsg::eventInputQueue(p,i,toQname);
    _flushQueue(_myOutputEvQueue[i] = _openQueue(toQname,q_attr));
  }

  { int pfd[2];
    if(::pipe(pfd)<0)
      perror("Opening pipe");

    _initFd         = pfd[1];
    _pfd[0].fd      = pfd[0];
    _pfd[0].events  = POLLIN;
    _pfd[0].revents = 0;
  }

  q_attr.mq_maxmsg  = _numberOfEvBuffers;
  q_attr.mq_msgsize = (long int)sizeof(XtcMonitorMsg);
  q_attr.mq_flags   = O_NONBLOCK;

  sprintf(toQname, "/PdsRequestQueue_%s",p);
  _requestQueue = _openQueue(toQname, q_attr);
  _flushQueue(_requestQueue);

  _pfd[1].fd = _myInputEvQueue;
  _pfd[1].events  = POLLIN;
  _pfd[1].revents = 0;

  q_attr.mq_maxmsg  = _numberOfEvBuffers;
  q_attr.mq_msgsize = (long int)sizeof(ShMsg);
  q_attr.mq_flags   = O_NONBLOCK;

  sprintf(toQname, "/PdsShuffleQueue_%s",p);
  _shuffleQueue = _openQueue(toQname, q_attr);
  { ShMsg m; _flushQueue(_shuffleQueue,(char*)&m, sizeof(m)); }

  _pfd[2].fd = _shuffleQueue;
  _pfd[2].events  = POLLIN;
  _pfd[2].revents = 0;

  // create the listening threads
  _terminate.store(false, std::memory_order_release);
  _taskThread = std::thread(&XtcMonitorServer::routine,  std::ref(*this));
  _discThread = std::thread(&XtcMonitorServer::discover, std::ref(*this));

  delete[] shmName;
  delete[] toQname;
  delete[] fromQname;

  return ret;
}

void XtcMonitorServer::_initialize_client()
{
#ifdef DBUG
  printf("Reading from %d\n",_pfd[0].fd);
#endif

  int s;
  if (::read(_pfd[0].fd,&s,sizeof(s))<0) {
    perror("Error reading client socket");
    abort();
  }

#ifdef DBUG
  printf("initialize client socket %d [%d]\n",s,_nfd);
#endif

  int iclient=-1;
  for(unsigned i=0; i<_myTrFd.size(); i++) {
    if (_myTrFd[i] == -1) {
      iclient = i;
      break;
    }
  }
  if (iclient == -1) {
    iclient = _myTrFd.size();
    if (iclient == int(_numberOfEvQueues)) {
      printf("Rejecting client %d : Number of EvQueues = %d\n",
             iclient, _numberOfEvQueues);
      ::close(s);
      return;
    }
    _myTrFd.push_back(-1);
  }

  _pfd[_nfd].fd = s;
  _pfd[_nfd].events  = POLLIN;
  _pfd[_nfd].revents = 0;
  _nfd++;

  _myTrFd[iclient] = s;
  printf("Initialized client %d [socket %d]\n",iclient,s);

  _myMsg.bufferIndex(iclient);

  if (::send(_myTrFd[iclient], (const char*)&_myMsg, sizeof(_myMsg), 0)<0) {
    perror("first send to client");
    abort();
  }

  _transitionCache->deallocate(iclient);
  _update(iclient,TransitionId::Reset);
}

//
//  Send the cached transitions to update the client's state to the current DAQ state
//
void XtcMonitorServer::_update(int iclient,
                               TransitionId::Value last)
{
  TransitionId::Value next = TransitionId::Value(last^1);
  std::stack<int> tr(_transitionCache->current());
  while(!tr.empty()) {
    int itr = tr.top(); tr.pop();
    int ib  = itr+_numberOfEvBuffers;
    if (reinterpret_cast<const Dgram*>(_myShm+_sizeOfBuffers*ib)->seq.service()>=next) {
      _myMsg.bufferIndex(ib);

      if (_transitionCache->allocate(itr,iclient))
        if (::send(_myTrFd[iclient], (const char*)&_myMsg, sizeof(_myMsg), 0)<0) {
          perror("Error sending current");
          _transitionCache->deallocate(itr,iclient);
        }
    }
  }
}

void XtcMonitorServer::_copyDatagram(Dgram* p, char* b)
{
  Dgram* dg = (Dgram*)p;
  memcpy((char*)b, dg, sizeof(Dgram)+dg->xtc.sizeofPayload());
}

void XtcMonitorServer::_deleteDatagram(Dgram* p, int) { _deleteDatagram(p); }

void XtcMonitorServer::_deleteDatagram(Dgram* p) {}

void XtcMonitorServer::_requestDatagram(int) { _requestDatagram(); }

void XtcMonitorServer::_requestDatagram() {}

mqd_t XtcMonitorServer::_openQueue(const char* name, mq_attr& attr)
{
  mqd_t q = mq_open(name,  O_CREAT|O_RDWR, PERMS, &attr);
  if (q == (mqd_t)-1) {
    perror("mq_open output");
    printf("name: '%s'\nmq_attr:\n\tmq_flags 0x%0lx\n\tmq_maxmsg 0x%0lx\n\tmq_msgsize 0x%0lx\n\t mq_curmsgs 0x%0lx\n",
           name, attr.mq_flags, attr.mq_maxmsg, attr.mq_msgsize, attr.mq_curmsgs );
    fprintf(stderr, "Initializing XTC monitor server encountered an error!\n");
    delete this;
    exit(EXIT_FAILURE);
  }
  else {  // Open twice to set all of the attributes
    printf("Opened queue %s (%d)\n",name,q);
  }

  mq_attr r_attr;
  mq_getattr(q,&r_attr);
  if (r_attr.mq_maxmsg != attr.mq_maxmsg ||
      r_attr.mq_msgsize!= attr.mq_msgsize) {

    printf("Failed to set queue attributes the first time.\n");
    mq_close(q);

    mqd_t q = mq_open(name,  O_CREAT|O_RDWR, PERMS, &attr);
    mq_getattr(q,&r_attr);

    if (r_attr.mq_maxmsg != attr.mq_maxmsg ||
        r_attr.mq_msgsize!= attr.mq_msgsize) {
      printf("Failed to set queue attributes the second time.\n");
      printf("open attr  %lx %lx %lx  read attr %lx %lx %lx\n",
             attr.mq_flags, attr.mq_maxmsg, attr.mq_msgsize,
             r_attr.mq_flags, r_attr.mq_maxmsg, r_attr.mq_msgsize);
    }
  }

  return q;
}

void XtcMonitorServer::_flushQueue(mqd_t q)
{
  XtcMonitorMsg m;
  _flushQueue(q,(char*)&m,sizeof(m));
}

void XtcMonitorServer::_flushQueue(mqd_t q, char* m, unsigned sz)
{
  // flush the queues just to be sure they are empty.
  struct mq_attr attr;
  do {
    mq_getattr(q, &attr);
    if (attr.mq_curmsgs)
      mq_timedreceive(q, m, sz, NULL, &_tmo);
  } while (attr.mq_curmsgs);
}

void XtcMonitorServer::_moveQueue(mqd_t iq, mqd_t oq)
{
  XtcMonitorMsg m;
  struct mq_attr attr;
  do {
    mq_getattr(iq, &attr);
    if (attr.mq_curmsgs) {
      if (mq_timedreceive(iq, (char*)&m, sizeof(m), NULL, &_tmo) == -1) {
        perror("moveQueue: mq_timedreceive");
        break;
      }
      else if (mq_timedsend   (oq, (char*)&m, sizeof(m), 0, &_tmo) == -1) {
        printf("Failed to reclaim buffer %i : %s\n",
               m.bufferIndex(), strerror(errno));
      }
      else
        _msgDest[m.bufferIndex()]=-1;
    }
  } while (attr.mq_curmsgs);
}

void XtcMonitorServer::unlink()
{
  _terminate.store(true, std::memory_order_release);

  //printf("Unlinking Message Queues... \n");
  mq_close(_myInputEvQueue);

  for(unsigned i=0; i<_numberOfEvQueues; i++) {
    mq_close(_myOutputEvQueue[i]);
  }
  mq_close(_requestQueue);
  mq_close(_shuffleQueue);
  mq_close(_discoveryQueue);

  char* qname = new char[128];
  for(unsigned i=0; i<_numberOfEvQueues; i++) {
    XtcMonitorMsg::eventInputQueue     (_tag,i,qname); mq_unlink(qname);
  }
  XtcMonitorMsg::eventInputQueue     (_tag,_numberOfEvQueues,qname); mq_unlink(qname);
  sprintf(qname, "/PdsRequestQueue_%s",_tag);  mq_unlink(qname);
  sprintf(qname, "/PdsShuffleQueue_%s",_tag);  mq_unlink(qname);
  XtcMonitorMsg::discoveryQueue      (_tag,qname); mq_unlink(qname);
  delete[] qname;
}

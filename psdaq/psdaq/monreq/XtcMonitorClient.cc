#include <stdlib.h>
#include <stdio.h>
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

#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "XtcMonitorClient.hh"
#include "XtcMonitorMsg.hh"

#include <poll.h>

//#define DBUG

enum {PERMS = S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH};
enum {PERMS_IN  = S_IRUSR|S_IRGRP|S_IROTH};
enum {PERMS_OUT  = S_IWUSR|S_IWGRP|S_IWOTH};
enum {OFLAGS = O_RDONLY};

static mqd_t _openQueue(const char* name, unsigned flags, unsigned perms,
                        bool lwait=true)
{
  struct mq_attr mymq_attr;
  mqd_t queue;
  while(1) {
    queue = mq_open(name, flags, perms, &mymq_attr);
    if (queue == (mqd_t)-1) {
      char b[128];
      sprintf(b,"mq_open %s",name);
      perror(b);
      sleep(1);
      if (!lwait) break;
    }
    else {
      printf("Opened queue %s (%d)\n",name,queue);
      break;
    }
  }
  return queue;
}

namespace Pds {
  namespace MonReq {
    class DgramHandler {
    public:
      DgramHandler(XtcMonitorClient& client,
                   int trfd, mqd_t evqin, mqd_t* evqout, unsigned ev_index,
                   const char* tag, char* myShm) :
        _client(client),
        _trfd(trfd), _evqin(evqin), _evqout(evqout), _ev_index(ev_index),
        _tag(tag), _shm(myShm), _last(XtcData::TransitionId::Reset)
      {
        _tmo.tv_sec = _tmo.tv_nsec = 0;
      }
      ~DgramHandler() {}
    public:
      enum Request { Continue, Reconnect, Return };
      Request transition() {
        XtcMonitorMsg myMsg;
        int nb = ::recv(_trfd, (char*)&myMsg, sizeof(myMsg), 0);
        if (nb < 0) {
          perror("transition receive");
          return Continue;
        }
        if (nb == 0) {
#ifdef DBUG
          printf("Received tr disconnect [%d]\n",_trfd);
#endif
          return Reconnect;
        }
        int i = myMsg.bufferIndex();
#ifdef DBUG
        printf("Received tr buffer %d [%d]\n",i,_trfd);
#endif
        if ( (i>=0) && (i<myMsg.numberOfBuffers())) {
          XtcData::Dgram* dg = (XtcData::Dgram*) (_shm + (myMsg.sizeOfBuffers() * i));
          _last = dg->seq.service();
          if (_client.processDgram(dg))
            return Return;
#ifdef DBUG
          printf("Returning tr buffer %d [%d]\n",i,_trfd);
#endif
          if (::send(_trfd,(char*)&myMsg,sizeof(myMsg),0)<0) {
            perror("transition send");
            return Return;
          }
        }
        else {
          fprintf(stderr, "ILLEGAL BUFFER INDEX %d\n", i);
          uint32_t* p = reinterpret_cast<uint32_t*>(&myMsg);
          fprintf(stderr, "XtcMonitorMsg: %x/%x/%x/%x [%d]\n",p[0],p[1],p[2],p[3],nb);
          return Return;
        }
        return Continue;
      }
      Request event     () {
        mqd_t  iq = _evqin;
        mqd_t* oq = _evqout;
        unsigned ioq = _ev_index;

        XtcMonitorMsg myMsg;
        unsigned priority(0);
        int nb;
        if ((nb=mq_receive(iq, (char*)&myMsg, sizeof(myMsg), &priority)) < 0) {
          perror("mq_receive buffer");
          return Return;
        }
        else {
          int i = myMsg.bufferIndex();
#ifdef DBUG
          printf("Received ev buffer %d [%d]\n",i,iq);
#endif
          if ( (i>=0) && (i<myMsg.numberOfBuffers())) {
            XtcData::Dgram* dg = (XtcData::Dgram*) (_shm + (myMsg.sizeOfBuffers() * i));
            if (_last==XtcData::TransitionId::Enable &&
                _client.processDgram(dg))
              return Return;
            if (oq==NULL)
              ;
            else if (myMsg.serial()) {
              while (mq_timedsend(oq[ioq], (const char *)&myMsg, sizeof(myMsg), priority, &_tmo)) {
                if (oq[++ioq]==-1) {
                  char qname[128];
                  XtcMonitorMsg::eventOutputQueue(_tag, ioq, qname);
                  oq[ioq] = _openQueue(qname, O_WRONLY, PERMS_OUT, false);
                }
              }
            }
            else {
              if (mq_timedsend(oq[0], (const char *)&myMsg, sizeof(myMsg), priority, &_tmo)) {
                ;
              }
            }
          }
          else {
            fprintf(stderr, "ILLEGAL BUFFER INDEX %d\n", i);
            uint32_t* p = reinterpret_cast<uint32_t*>(&myMsg);
            fprintf(stderr, "XtcMonitorMsg: %x/%x/%x/%x [%d]\n",p[0],p[1],p[2],p[3],nb);
            return Return;
          }
        }
        return Continue;
      }
    private:
      XtcMonitorClient&            _client;
      int                          _trfd;
      mqd_t                        _evqin;
      mqd_t*                       _evqout;
      unsigned                     _ev_index;
      const char*                  _tag;
      char*                        _shm;
      timespec                     _tmo;
      XtcData::TransitionId::Value _last;
    };
  };
};

using namespace XtcData;
using namespace Pds::MonReq;

int XtcMonitorClient::processDgram(Dgram* dg) {
  printf("%-15s transition: time 0x%014lx, payloadSize 0x%x\n",
         TransitionId::name(dg->seq.service()),
         dg->seq.pulseId().value(),dg->xtc.sizeofPayload());
  return 0;
}

int XtcMonitorClient::run(const char* tag, int tr_index)
{ return run(tag, tr_index, tr_index); }

int XtcMonitorClient::run(const char* tag, int tr_index, int) {
  int error = 0;
  char* qname             = new char[128];

  umask(0);   // Need this to set group/other write permissions on mqueue

  XtcMonitorMsg myMsg;
  unsigned priority;

  mqd_t* myOutputEvQueues = 0;

  //
  //  Request initialization
  //

  while(1) {
    int myTrFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (myTrFd < 0) {
      perror("Opening myTrFd socket");
      return 1;
    }

    while(1) {
      XtcMonitorMsg::discoveryQueue(tag,qname);
      mqd_t discoveryQueue = _openQueue(qname, O_RDONLY, PERMS_IN);
      if (discoveryQueue == (mqd_t)-1)
	error++;

      if (mq_receive(discoveryQueue, (char*)&myMsg, sizeof(myMsg), &priority) < 0) {
	perror("mq_receive discoveryQ");
	return ++error;
      }

      mq_close(discoveryQueue);

      sockaddr_in saddr;
      saddr.sin_family = AF_INET;
      saddr.sin_addr.s_addr = htonl(0x7f000001);
      saddr.sin_port        = htons(myMsg.bufferIndex());

      if (::connect(myTrFd, (sockaddr*)&saddr, sizeof(saddr)) < 0) {
	perror("Connecting myTrFd socket");
	sleep(1);
      }
      else {
#ifdef DBUG
        socklen_t addrlen = sizeof(sockaddr_in);
        sockaddr_in name;
        ::getsockname(myTrFd, (sockaddr*)&name, &addrlen);
	printf("Connected to %08x.%d [%d] from %08x.%d\n",
               ntohl(saddr.sin_addr.s_addr),ntohs(saddr.sin_port),myTrFd,
               ntohl(name.sin_addr.s_addr),ntohs(name.sin_port));
#endif
	break;
      }
    }

    if (::read(myTrFd,&myMsg,sizeof(myMsg))!=sizeof(myMsg)) {
      printf("Connection rejected by shmem server [too many clients]\n");
      return 1;
    }

    //
    //  Initialize shared memory from first message
    //
    unsigned sizeOfShm = myMsg.numberOfBuffers() * myMsg.sizeOfBuffers();
    unsigned pageSize  = (unsigned)sysconf(_SC_PAGESIZE);
    unsigned remainder = sizeOfShm % pageSize;
    if (remainder)
      sizeOfShm += pageSize - remainder;

    XtcMonitorMsg::sharedMemoryName(tag, qname);
    printf("Opening shared memory %s of size 0x%x (0x%x * 0x%x)\n",
	   qname,sizeOfShm,myMsg.numberOfBuffers(),myMsg.sizeOfBuffers());

    int shm = shm_open(qname, OFLAGS, PERMS_IN);
    if (shm < 0) perror("shm_open");
    char* myShm = (char*)mmap(NULL, sizeOfShm, PROT_READ, MAP_SHARED, shm, 0);
    if (myShm == MAP_FAILED) perror("mmap");
    else printf("Shared memory at %p\n", (void*)myShm);

    int ev_index = myMsg.bufferIndex();
    XtcMonitorMsg::eventInputQueue(tag,ev_index,qname);
    mqd_t myInputEvQueue = _openQueue(qname, O_RDONLY, PERMS_IN);
    if (myInputEvQueue == (mqd_t)-1)
      error++;

    myOutputEvQueues = new mqd_t[myMsg.numberOfQueues()+1];
    for(int i=0; i<=myMsg.numberOfQueues(); i++)
      myOutputEvQueues[i]=-1;

    if (myMsg.serial()) {
      XtcMonitorMsg::eventOutputQueue(tag,ev_index,qname);
      myOutputEvQueues[ev_index] = _openQueue(qname, O_WRONLY, PERMS_OUT);
      if (myOutputEvQueues[ev_index] == (mqd_t)-1)
	error++;
    }
    else {
      XtcMonitorMsg::eventInputQueue(tag,myMsg.return_queue(),qname);
      myOutputEvQueues[0] = _openQueue(qname, O_WRONLY, PERMS_OUT);
      if (myOutputEvQueues[0] == (mqd_t)-1)
	error++;
    }

    if (error) {
      fprintf(stderr, "Could not open at least one message queue!\n");
      fprintf(stderr, "tag %s, tr_index %d, ev_index %d\n",tag,tr_index,ev_index);
      return error;
    }


    //
    //  Seek the Map transition
    //
    do {
      if (::recv(myTrFd, (char*)&myMsg, sizeof(myMsg), MSG_WAITALL) < 0) {
	perror("mq_receive buffer");
	return ++error;
      }
      else {
	int i = myMsg.bufferIndex();
	if ( (i>=0) && (i<myMsg.numberOfBuffers())) {
	  Dgram* dg = (Dgram*) (myShm + (myMsg.sizeOfBuffers() * i));
	  if (dg->seq.service()==TransitionId::Map) {
	    if (!processDgram(dg)) {
	      if (::send(myTrFd,(char*)&myMsg,sizeof(myMsg),0)<0) {
		perror("transition send");
		return false;
	      }
	      break;
	    }
          }
          else
            printf("Unexpected transition %s != Map\n",TransitionId::name(dg->seq.service()));
	}
        else
          printf("Illegal transition buffer index %d\n",i);
      }
    } while(1);

    //
    //  Handle all transitions first, then events
    //
    pollfd pfd[2];
    pfd[0].fd      = myTrFd;
    pfd[0].events  = POLLIN | POLLERR;
    pfd[0].revents = 0;
    pfd[1].fd      = myInputEvQueue;
    pfd[1].events  = POLLIN | POLLERR;
    pfd[1].revents = 0;
    int nfd = 2;

    DgramHandler handler(*this,
			 myTrFd,
			 myInputEvQueue,myOutputEvQueues,ev_index,
			 tag,myShm);

    DgramHandler::Request r=DgramHandler::Continue;
    while (r==DgramHandler::Continue) {
      if (::poll(pfd, nfd, -1) > 0) {
	if (pfd[0].revents & POLLIN) { // Transition
	  r = handler.transition();
	}
	else if (pfd[1].revents & POLLIN) { // Event
	  r = handler.event     ();
	}
      }
    }

    if (myOutputEvQueues)
      delete[] myOutputEvQueues;

    close(myTrFd);

    if (r==DgramHandler::Return) return 1;
  }
}


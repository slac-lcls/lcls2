#include "psdaq/cphw/Reg.hh"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <unistd.h>
#include <poll.h>
#include <stdlib.h>
#include <semaphore.h>

//#define DBUG

static unsigned _fd;
static unsigned _context=0;
static unsigned _mem;
static sem_t    _sem;

using namespace Pds::Cphw;

void Reg::setBit  (unsigned b)
{
  unsigned r = *this;
  *this = r | (1<<b);
}

void Reg::clearBit(unsigned b)
{
  unsigned r = *this;
  *this = r &~(1<<b);
}

void Reg::set(const char* host,
              unsigned short port,
              unsigned mem,
              unsigned long long memsz)
{
  unsigned ip = ntohl(inet_addr(host));
  _fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in saddr;
  saddr.sin_family      = AF_INET;
  saddr.sin_addr.s_addr = htonl(ip);
  saddr.sin_port        = htons(port);
  socklen_t addrlen = sizeof(saddr);
  ::connect(_fd, reinterpret_cast<const sockaddr*>(&saddr), addrlen);
  _mem = mem;

  sem_init(&_sem,0,1);
}

Reg& Reg::operator=(const unsigned r)
{
  uint32_t addr = ((_mem+uint64_t(this))>>2)&0x3fffffff; 
  uint32_t tbuff[4];
  tbuff[0] = _context;
  tbuff[1] = addr | 0x40000000;
  tbuff[2] = r;
  tbuff[3] = 0;

  int tmo = 20;

  sem_wait(&_sem);

  for(unsigned i=0; i<10; i++) {
#ifdef DBUG
    printf("Send %08x %08x %08x %08x\n",
           tbuff[0],tbuff[1],tbuff[2],tbuff[3]);
#endif
    ::send(_fd, tbuff, sizeof(tbuff), 0);

    pollfd pfd;
    pfd.fd = _fd;
    pfd.events = POLLIN | POLLERR;

    int pv = poll(&pfd, 1, tmo);
    if (pv < 0) {
      printf("Reg: Error on socket %d\n",_fd);
      exit(1);
    }

    if (pv==0) {
      fprintf(stderr,"[Reg: timeout %d]",i+1);
      continue;
    }

    uint32_t rbuff[4];
    int ret = ::read(_fd, rbuff, sizeof(rbuff));
#ifdef DBUG
    printf("Recv %08x %08x %08x %08x [%d]\n",
	 rbuff[0],rbuff[1],rbuff[2],rbuff[3],ret);
#endif
    if (ret      != sizeof(rbuff) ||
	rbuff[0] != _context ||
	(rbuff[1]^(1<<30)) != addr) {
      printf("Error reading [%x,%x]/[%x,%x] %d bytes, retry %d\n",
	     rbuff[0], rbuff[1], _context, addr, ret, i+1);
      continue;
    }

    if (rbuff[2] != r) {
      printf("Ack error %08x:%08x\n",rbuff[2],r);
      continue;
    }
    sem_post(&_sem);
    return *this;
  }

  printf("Reg[%p]::operator=[%08x]  FAILED\n",this,r);
  sem_post(&_sem);
  return *this;
}

Reg::operator unsigned() const
{
  // semaphore
  // sendmsg to read register
  // recvmsg for value
  // return value
  uint32_t addr = ((_mem+uint64_t(this))>>2)&0x3fffffff; 
  uint32_t tbuff[4];
  tbuff[0] = _context;
  tbuff[1] = addr;
  tbuff[2] = 0;
  tbuff[3] = 0;

  int tmo = 20;

  sem_wait(&_sem);

  for(unsigned i=0; i<10; i++) {
#ifdef DBUG
    printf("Send %08x %08x %08x %08x\n",
	   tbuff[0],tbuff[1],tbuff[2],tbuff[3]);
#endif
    ::send(_fd, tbuff, sizeof(tbuff), 0);

    pollfd pfd;
    pfd.fd = _fd;
    pfd.events = POLLIN | POLLERR;
    
    int pv = poll(&pfd, 1, tmo);
    if (pv < 0) {
      printf("Reg: Error on socket %d\n",_fd);
      exit(1);
    }

    if (pv==0) {
      fprintf(stderr,"[Reg: timeout %d]",i+1);
      continue;
    }

    uint32_t rbuff[4];
    int ret = ::read(_fd, rbuff, sizeof(rbuff));
#ifdef DBUG
    printf("Recv %08x %08x %08x %08x [%d]\n",
	   rbuff[0],rbuff[1],rbuff[2],rbuff[3],ret);
#endif

    if (ret      != sizeof(rbuff) ||
	rbuff[0] != _context ||
	rbuff[1] != addr) {
      printf("Error reading [%x,%x]/[%x,%x] %d bytes, retry %d\n",
	     rbuff[0], rbuff[1], _context, addr, ret, i+1);
      continue;
    }
  
    sem_post(&_sem);
    return rbuff[2];
  }

  printf("Reg[%p]::operator unsigned() FAILED\n",this);
  sem_post(&_sem);
  return 0;
}

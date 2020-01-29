#include "ModuleBase.hh"
#include <netdb.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>

using Pds::HSD::ModuleBase;

ModuleBase* ModuleBase::create(int fd)
{
  void* ptr = mmap(0, sizeof(ModuleBase), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map");
    return 0;
  }

  ModuleBase* m = new(ptr) ModuleBase;

  Pds::Mmhw::RegProxy::initialize(m, m->regProxy);
  
  return m;
}

void ModuleBase::setRxAlignTarget(unsigned t)
{
  unsigned v = gthAlignTarget;
  v &= ~0x3f;
  v |= (t&0x3f);
  gthAlignTarget = v;
}

void ModuleBase::setRxResetLength(unsigned len)
{
  unsigned v = gthAlignTarget;
  v &= ~0xf0000;
  v |= (len&0xf)<<16;
  gthAlignTarget = v;
}
 
void ModuleBase::dumpRxAlign     () const
{
  printf("\nTarget: %u\tRstLen: %u\tLast: %u\n",
         gthAlignTarget&0x7f,
         (gthAlignTarget>>16)&0xf, 
         gthAlignLast&0x7f);
  for(unsigned i=0; i<128; i++) {
    printf(" %04x",(gthAlign[i/2] >> (16*(i&1)))&0xffff);
    if ((i%10)==9) printf("\n");
  }
  printf("\n");
}

unsigned ModuleBase::local_id(unsigned bus)
{
  struct addrinfo hints;
  struct addrinfo* result;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_INET;       /* Allow IPv4 or IPv6 */
  hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
  hints.ai_flags = AI_PASSIVE;    /* For wildcard IP address */

  char hname[64];
  gethostname(hname,64);
  int s = getaddrinfo(hname, NULL, &hints, &result);
  if (s != 0) {
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
    exit(EXIT_FAILURE);
  }

  sockaddr_in* saddr = (sockaddr_in*)result->ai_addr;

  unsigned id = 0xfc000000 | (bus<<16) |
    (ntohl(saddr->sin_addr.s_addr)&0xffff);

  return id;
}

//
//  Test for unicast/multicast transmit/receive
//
#include "AppUtils.hh"

#include <sys/types.h>
#include <sys/socket.h>

#include <arpa/inet.h>
#include <time.h>

#include <sys/ioctl.h>
#include <net/if.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

#include <vector>

static void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("\t-i\tNetwork interface: name or address\n");
  printf("\t-a\tDestination address, dotted notation\n");
  printf("\t-p\tDestination port\n");
  printf("\t-s\tBuffer size, bytes\n");
  printf("\t-r\tReceive mode\n");
  printf("\t-P\tPeek at data\n");
  printf("\t-d\tNumber of packets to dump\n");
  printf("\t-D\tDelay between tmit packets [usec]\n");
}

int main(int argc, char **argv) 
{
  unsigned interface = 0x7f000001;
  unsigned port  = 0;
  ssize_t  sz    = 0x2000;
  unsigned ndump = 0;
  bool lreceiver = false;
  bool lpeek = false;
  std::vector<unsigned> uaddr;
  unsigned tdelay = 0;
  bool checkfid = false;

  char c;
  while ( (c=getopt( argc, argv, "i:a:d:D:p:s:frPh?")) != EOF ) {
    switch(c) {
    case 'a': 
      for(char* arg = strtok(optarg,","); arg!=NULL; arg=strtok(NULL,","))
	uaddr.push_back(Psdaq::AppUtils::parse_ip(arg));
      break;
    case 'i':
      interface = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case 'p':
      port = strtoul(optarg,NULL,0);
      break;
    case 'd':
      ndump = strtoul(optarg,NULL,0);
      break;
    case 'D':
      tdelay = strtoul(optarg,NULL,0);
      break;
    case 's':
      sz   = strtoul(optarg,NULL,0);
      break;
    case 'r':
      lreceiver = true;
      break;
    case 'f':
      checkfid = true;
      break;
    case 'P':
      lpeek = true;
      break;
    default:
      usage(argv[0]);
      return 1;
    }
  }

  std::vector<sockaddr_in> address(uaddr.size());
  for(unsigned i=0; i<uaddr.size(); i++) {
    sockaddr_in sa;
    sa.sin_family      = AF_INET;
    sa.sin_addr.s_addr = htonl(uaddr[i]);
    sa.sin_port        = htons(port);
    memset(sa.sin_zero,0,sizeof(sa.sin_zero));
    address[i] = sa;
    printf("dst[%d] : %x.%d\n",i,uaddr[i],port);
  }

  int fd;
  if ((fd = ::socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("socket");
    return -1;
  }
  
  unsigned skb_size = 0x100000;

  int y=1;
  if(setsockopt(fd, SOL_SOCKET, SO_BROADCAST, (char*)&y, sizeof(y)) == -1) {
    perror("set broadcast");
    return -1;
  }
      
  if (lreceiver) {
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, 
                     (char*)&skb_size, sizeof(skb_size)) < 0) {
      perror("so_rcvbuf");
      return -1;
    }

    if (::bind(fd, (sockaddr*)&address[0], sizeof(sockaddr_in)) < 0) {
      perror("bind");
      return -1;
    }
    
    if ((uaddr[0]>>28) == 0xe) {

      if(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char*)&y, sizeof(y)) == -1) {
	perror("set reuseaddr");
	return -1;
      }

      struct ip_mreq ipMreq;
      bzero ((char*)&ipMreq, sizeof(ipMreq));
      ipMreq.imr_multiaddr.s_addr = htonl(uaddr[0]);
      ipMreq.imr_interface.s_addr = htonl(interface);
      int error_join = setsockopt (fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, 
                                   (char*)&ipMreq, sizeof(ipMreq));
      if (error_join==-1) {
        perror("mcast_join");
        return -1;
      }
      else
	printf("Joined %x on if %x\n", uaddr[0], interface);
    }
  }
  else {
    if (::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, 
                     (char*)&skb_size, sizeof(skb_size)) < 0) {
      perror("so_sndbuf");
      return -1;
    }

    { sockaddr_in sa;
      sa.sin_family      = AF_INET;
      sa.sin_addr.s_addr = htonl(interface);
      sa.sin_port        = htons(port);
      memset(sa.sin_zero,0,sizeof(sa.sin_zero));
      printf("binding to %x.%d\n",ntohl(sa.sin_addr.s_addr),ntohs(sa.sin_port));
      if (::bind(fd, (sockaddr*)&sa, sizeof(sockaddr_in)) < 0) {
        perror("bind");
        return -1;
      }
      sockaddr_in name;
      socklen_t name_len=sizeof(name);
      if (::getsockname(fd,(sockaddr*)&name,&name_len) == -1) {
        perror("getsockname");
        return -1;
      }
      printf("bound to %x.%d\n",ntohl(name.sin_addr.s_addr),ntohs(name.sin_port));
    }

    if ((uaddr[0]>>28) == 0xe) {
      in_addr addr;
      addr.s_addr = htonl(interface);
      if (setsockopt(fd, IPPROTO_IP, IP_MULTICAST_IF, (char*)&addr,
                     sizeof(in_addr)) < 0) {
        perror("set ip_mc_if");
        return -1;
      }
    }
  }

  uint64_t tbytes = 0;
  uint64_t npkts = 0;

  Psdaq::MonitorArgs args;
  args.add("Packets","Pkts",npkts);
  args.add("Bytes","B",tbytes);
  Psdaq::AppUtils::monitor(args);

  char* buff = new char[sz];
  iphdr* ip = (iphdr*)buff;
  ip->ihl = 5;
  ip->version = 4;
  ip->tos = 0;
  ip->tot_len = sz + sizeof(udphdr) + sizeof(iphdr);
  ip->id = 0;
  ip->frag_off = 0;
  ip->ttl = 1;
  ip->protocol = IPPROTO_UDP;
  ip->check = 0;
  ip->saddr = htonl(interface|0xff);

  udphdr* udp = (udphdr*)(ip+1);
  udp->source = 0;
  udp->dest   = htons(port);
  udp->len    = htons(sz+sizeof(udphdr));
  udp->check  = 0;

  unsigned iaddr=0;
  
  unsigned fido=0;

  if (lreceiver) {
      sockaddr_in src;
      socklen_t len = sizeof(src);
      ::recvfrom(fd, buff, sz, 0, (sockaddr*)&src, &len);
      unsigned srcip = ntohl(src.sin_addr.s_addr);
      unsigned sport = ntohs(src.sin_port);
      printf("received from %u.%u.%u.%u:%u\n",
             (srcip>>24)&0xff,
             (srcip>>16)&0xff,
             (srcip>> 8)&0xff,
             (srcip>> 0)&0xff,
             sport);
  }

  while(1) {

    ssize_t bytes;
    if (lreceiver) {
      if (lpeek) {
        int nbsz = 32;
        int nb = ::recv(fd, buff, nbsz, MSG_PEEK);
        if (nb != nbsz) {
          printf("peek %d/%u\n",nb,nbsz);
        }
      }
      bytes = ::recv(fd, buff, sz, 0);
      npkts++;
      const uint32_t* p = reinterpret_cast<const uint32_t*>(buff);
      if (ndump) {
        for(unsigned i=0; i<(bytes>>2); i++)
          printf("%08x%c",p[i],(i%8)==7?'\n':' ');
        printf("\n");
        --ndump;
      }
      else if (checkfid) {
        unsigned fid = p[3]&0x1ffff;
        int diff = (fido > fid) ? fid + 0x1ffe0 - fido : fid-fido;
        if (diff != 3)
          printf("fid [0x%05x] ofid [0x%05x] diff %u\n", fid, fido, diff);
        fido = fid;
      }
    }
    else {
      const sockaddr_in& sa = address[iaddr];
      bytes = ::sendto(fd, buff, sz, 0, (const sockaddr*)&sa, sizeof(sockaddr_in));
      iaddr = (iaddr+1)%address.size();
      if (tdelay)
        usleep(tdelay);
    }

    if (bytes < 0) {
      perror("recv/send");
      continue;
    }
    tbytes += bytes;
  }

  return 0;
}

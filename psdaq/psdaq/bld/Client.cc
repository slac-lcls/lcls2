#include "Client.hh"
#include "Header.hh"

#include <arpa/inet.h>
#include <string.h>
#include <string>
#include <unistd.h>

//#define DBUG

using namespace Pds::Bld;

Client::Client(unsigned interface,
               unsigned mcaddr,
               unsigned port)
{
  //
  //  Open the socket to receive multicasts
  //
  _fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (_fd < 0) {
    perror("Open socket");
    throw std::string("Open socket");
  }

  unsigned skb_size = 0x1000000;

  if (::setsockopt(_fd, SOL_SOCKET, SO_RCVBUF, 
                   (char*)&skb_size, sizeof(skb_size)) < 0) {
    perror("so_rcvbuf");
    throw std::string("Failed to setsockopt SO_RCVBUF");
  }

  sockaddr_in saddr;
  saddr.sin_family = AF_INET;
  saddr.sin_addr.s_addr = htonl(mcaddr);
  saddr.sin_port        = htons(port);
  memset(saddr.sin_zero,0,sizeof(saddr.sin_zero));

  if (::bind(_fd, (sockaddr*)&saddr, sizeof(saddr)) < 0) {
    perror("bind");
    throw std::string("bind");
  }
    
  int y = 1;
  if(setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, (const void*)&y, sizeof(y)) == -1) {
    perror("set reuseaddr");
    throw std::string("set reuseaddr");
  }

  struct ip_mreq ipMreq;
  bzero ((char*)&ipMreq, sizeof(ipMreq));
  ipMreq.imr_multiaddr.s_addr = htonl(mcaddr);
  ipMreq.imr_interface.s_addr = htonl(interface);
  int error_join = setsockopt (_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, 
                               (char*)&ipMreq, sizeof(ipMreq));
  if (error_join==-1) {
    perror("mcast_join");
    throw std::string("mcast_join");
  }
  else
    printf("Joined %x on if %x\n", mcaddr, interface);

  _buffer = new char[Header::MTU];
  _buffer_size = 0;
  _buffer_next = 0;
}

Client::~Client()
{
  delete[] _buffer;
  close(_fd);
}

//
//  Fetch the next contribution 
//  Return pulseId
//
uint64_t Client::fetch(char* payload, unsigned sizeofT)
{
  uint64_t result;
  //
  //  Check if the buffer is empty
  //
  if (_buffer_next + sizeofT > _buffer_size) {
    //
    //  Receive new multicast
    //
    ssize_t sz = ::recv(_fd, _buffer, Header::MTU, 0);
    if (sz < 0) {
      perror("Error receiving multicast");
      exit(1);
    }

    Header& h = *new (_buffer) Header;
    //  Check for ID change
    //  If so, drop the whole packet
    if (h.id() != _id)
      return 0;

    result = h.pulseId();
    memcpy(payload, _buffer+Header::sizeofFirst, sizeofT);
    _buffer_size = sz;
    _buffer_next = Header::sizeofFirst + sizeofT;

#ifdef DBUG
    printf("New mcast sz %u bytes  pid 0x%llx\n", sz, result);
    const unsigned* p = reinterpret_cast<const unsigned*>(_buffer);
    for(unsigned i=0; i<32; i++)
      printf("%08x%c",p[i],(i&7)==7 ? '\n':' ');
#endif
  }
  //
  //  Fill the payload with the next entry
  //
  else {
    result = (new (_buffer) Header)->pulseId();
    result += *reinterpret_cast<uint32_t*>(_buffer+_buffer_next)>>20;
    memcpy(payload, _buffer+_buffer_next+Header::sizeofNext, sizeofT);
    _buffer_next += Header::sizeofNext + sizeofT;
  }
  return result;
}

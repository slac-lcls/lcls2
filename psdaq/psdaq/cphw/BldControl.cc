#include "BldControl.hh"
#include "Utils.hh"
#include <new>
#include <stdio.h>

using namespace Pds::Cphw;

BldControl* BldControl::locate()
{
  return new ((void*)0) BldControl;
}

unsigned BldControl::maxSize    () const { return getf(maxSize_enable,11, 0); }
unsigned BldControl::packetCount() const { return getf(pktCount_pause,20, 0); }
unsigned BldControl::wordsLeft  () const { return getf(count_state   ,11, 0); }
unsigned BldControl::state      () const { return getf(count_state   , 4,16); }
unsigned BldControl::paused     () const { return getf(pktCount_pause, 1,31); }

void BldControl::setMaxSize(unsigned v) { setf(maxSize_enable, v, 11, 0); }

void BldControl::enable (int fd,
                         const sockaddr_in& sa) 
{ 
  // "connect" to the sending socket
  char buf[4];
  send(fd,buf,sizeof(buf),0);

  setf(maxSize_enable, 1, 1, 31);

  printf("Enabled (%x) on port(%x), addr(%x)\n", 
         getf(maxSize_enable,1,31),
         ntohs(getf(port,16,0)),
         ntohl(unsigned(ip)));
}

void BldControl::enable (const sockaddr_in& sa) 
{ 
  port = sa.sin_port;
  ip   = sa.sin_addr.s_addr;
  setf(maxSize_enable, 1, 1, 31);

  printf("Enabled (%x) on port(%x), addr(%x)\n", 
         getf(maxSize_enable,1,31),
         ntohs(getf(port,16,0)),
         ntohl(unsigned(ip)));
}

void BldControl::disable() 
{
  setf(maxSize_enable, 0, 1, 31);
  // port = 0;
  // ip   = 0;
}

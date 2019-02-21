#ifndef Pds_Cphw_BldControl_hh
#define Pds_Cphw_BldControl_hh

#include "psdaq/cphw/Reg.hh"
#include "psdaq/cphw/Reg64.hh"
#include "psdaq/cphw/AmcTiming.hh"
#include <arpa/inet.h>

namespace Pds {
  namespace Cphw {
    class BldControl {
    public:
      static class BldControl* locate();
    public:
      unsigned maxSize    () const;
      unsigned packetCount() const;
      unsigned wordsLeft  () const;
      unsigned state      () const;
      unsigned paused     () const;
    public:
      void setMaxSize(unsigned v);
      void enable (int fd,
                   const sockaddr_in& sa);
      void enable (const sockaddr_in& sa);
      void disable();
    public:
      Pds::Cphw::AmcTiming  _timing;
    private:
      char  _rsvd_0[0x09030000-sizeof(_timing)];       // 09030000
      Reg   maxSize_enable;
    public:
      Reg   channelMask;
      Reg64 channelSevr;
    private:
      Reg   count_state;
    public:
      Reg   pulseIdL;
      Reg   timeStampL;
      Reg   delta;      
    private:
      Reg   pktCount_pause;
    private:
      char  _rsvd_09030024[(0x0a000828-0x09030024)];
      Reg   port;
      Reg   ip;
    protected:
      BldControl() {}
    };
  };
};

#endif


#ifndef Pds_Bld_Client_hh
#define Pds_Bld_Client_hh

#include <stdint.h>

namespace Pds {
  namespace Bld {
    class Client {
    public:
      Client(unsigned interface,
             unsigned mcaddr,
             unsigned port);
      ~Client();
    public:
      //
      //  Fetch the next contribution 
      //  Return pulseId
      //
      uint64_t fetch(char* payload, unsigned sizeofT);
    private:
      int      _fd;
      char*    _buffer;
      unsigned _buffer_size;
      unsigned _buffer_next;
    };
  };
};

#endif

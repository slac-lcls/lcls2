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
      void     setID(unsigned v) { _id=v; }
      unsigned getID() const { return _id; }
      //
      //  Fetch the next contribution 
      //  Return pulseId or 0 if ID has changed
      //
      uint64_t fetch(char* payload, unsigned sizeofT);
    private:
      int      _fd;
      unsigned _id;
      char*    _buffer;
      unsigned _buffer_size;
      unsigned _buffer_next;
    };
  };
};

#endif

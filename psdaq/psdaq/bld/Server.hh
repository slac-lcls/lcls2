#ifndef Pds_Bld_Server_hh
#define Pds_Bld_Server_hh

#include <stdint.h>

namespace Pds {
  namespace Bld {
    class Server {
    public:
      Server(int fd);
      ~Server();
    public:
      unsigned id() const { return _id; }
      void setID  (uint32_t    id);
      void publish(uint64_t    pulseId,
                   uint64_t    timeStamp,
                   const char* T,
                   unsigned    sizeofT);
      void flush  ();
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

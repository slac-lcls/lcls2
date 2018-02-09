#ifndef BldServer_hh
#define BldServer_hh

#include <stdint.h>

namespace Bld {
  template <class T>
  class BldServer {
  public:
    BldServer(int fd, unsigned src);
    ~BldServer();
  public:
    void publish(uint64_t pulseId,
                 uint64_t timeStamp,
                 const T&);
  private:
    int      _fd;
    unsigned _src;
    char*    _buffer;
    unsigned _buffer_size;    
    unsigned _buffer_next;
  };
};

#endif

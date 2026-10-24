/**
 * @file
 * @brief Pds::Bld::Server, which packs payloads into BLD multicast packets and sends them.
 */
#ifndef Pds_Bld_Server_hh
#define Pds_Bld_Server_hh

#include <stdint.h>

namespace Pds {
  namespace Bld {
    /** Packs payloads with BLD headers into an 8192-byte buffer and sends it as one packet when it is full or its first entry is old. Used by hpsBldServer and hpsBldCopy in psdaq/app. */
    class Server {
    public:
      /** Use socket fd for sending (with send(), so it must be connected); the ID starts at 0 and the buffer is empty. */
      Server(int fd);
      /** Does nothing; the buffer is not freed. */
      ~Server();
    public:
      /** Return the source ID. */
      unsigned id() const { return _id; }
      /** Set the source ID written into the full header of each packet. */
      void setID  (uint32_t    id);
      /** Append a header (full for the first entry of a packet, short otherwise) and the sizeofT-byte payload T. Send the packet if another payload of this size would not fit or pulseId exceeds the packet's first pulse ID by more than 1023. */
      void publish(uint64_t    pulseId,
                   uint64_t    timeStamp,
                   const char* T,
                   unsigned    sizeofT);
      /** Send the buffered packet, if any. */
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

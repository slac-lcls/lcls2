/**
 * @file
 * @brief Pds::Bld::Client, a receiver of BLD multicast packets.
 */
#ifndef Pds_Bld_Client_hh
#define Pds_Bld_Client_hh

#include <stdint.h>

namespace Pds {
  /** Namespace of the BLD multicast packet classes (Header, Client, Server). */
  namespace Bld {
    /** Receives BLD multicast packets and returns their payloads one entry at a time; a packet whose source ID differs from getID() is dropped. Used by hpsBldClient, hpsBldStat and hpsBldCopy in psdaq/app. */
    class Client {
    public:
      /** Open a UDP socket with a 16 MiB receive buffer, bind it to mcaddr and port and join multicast group mcaddr on interface (addresses in host byte order). Throws a std::string on failure; the expected ID is not initialized. */
      Client(unsigned interface,
             unsigned mcaddr,
             unsigned port);
      /** Free the packet buffer and close the socket. */
      ~Client();
    public:
      /** Set the expected source ID. */
      void     setID(unsigned v) { _id=v; }
      /** Return the expected source ID. */
      unsigned getID() const { return _id; }
      //
      //  Fetch the next contribution 
      //  Return pulseId or 0 if ID has changed
      //
      /** Copy the next sizeofT-byte payload into payload and return its pulse ID (Header::pulseId() of the packet, plus the 12-bit offset for later entries). Receives a new packet when the current one has no complete entry left; returns 0 if that packet's ID differs from getID() and exits the process on a receive error. */
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

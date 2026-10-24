/**
 * @file
 * @brief Pds::Bld::Header, the header layout of BLD multicast packets.
 */
#ifndef Pds_Bld_Header_hh
#define Pds_Bld_Header_hh

#include <stdio.h>

namespace Pds {
  namespace Bld {
    /** Header of a BLD multicast packet, overlaid on the packet buffer: a full header before the first payload and a short header, holding offsets from the full header, before each later payload. */
    class Header {
    public:
      /** Header sizes in bytes. */
      enum { sizeofFirst = 28, /**< Bytes before the first payload of a packet (28). */ sizeofNext = 12  /**< Bytes before each later payload (12); the short-header constructor writes only the first 4. */ };
      /** Packet buffer size. */
      enum { MTU = 8192  /**< Size in bytes of the packet buffers of Client and Server (8192). */ };
      /** Does nothing; used to overlay an existing buffer. */
      Header() {}
      /** Write a full header: pulseId in 64-bit word 0, timeStamp in 64-bit word 1 and src in the following 32-bit word; only these 20 bytes are written. */
      Header(uint64_t pulseId, uint64_t timeStamp, unsigned src)
      {
        uint64_t* p = reinterpret_cast<uint64_t*>(this);
        p[0] = pulseId;
        p[1] = timeStamp;
        *reinterpret_cast<uint32_t*>(&p[2]) = src;
      }
      /** Write a short header word relative to the full header ref: bits 31-20 hold the low 12 bits of the pulse ID difference and bits 19-0 the low 20 bits of the time stamp difference. */
      Header(uint64_t pulseId, uint64_t timeStamp, const Header& ref)
      {
        uint32_t* p = reinterpret_cast<uint32_t*>(this);
        const uint64_t* q = reinterpret_cast<const uint64_t*>(&ref);
        *p  = ((pulseId   - q[0])&0xfff)<<20;
        *p |= (timeStamp - q[1])&0xfffff;
      }
    public:
      /** Return true if pulseId exceeds 64-bit word 0 of this header by more than 1023. */
      bool done(uint64_t pulseId) const
      {
        return (pulseId - *reinterpret_cast<const uint64_t*>(this) > 1023);
      }
      /** Return 64-bit word 1 of the header. Note that the full-header constructor stores timeStamp there and pulseId in word 0. */
      uint64_t pulseId() const
      {
        return reinterpret_cast<const uint64_t*>(this)[1];
      }
      /** Return 32-bit word 4 (bytes 16-19), where the full-header constructor stores src. */
      unsigned id() const
      {
        return reinterpret_cast<const unsigned*>(this)[4];
      }
    private:
    };
  };
};

#endif

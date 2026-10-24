/**
 * @file
 * @brief Event-builder helpers (page rounding, page-aligned allocation, logging the negotiated libfabric configuration) and the ImmData packing of the 32-bit RDMA immediate data word.
 */
#ifndef Pds_Eb_Utilities_hh
#define Pds_Eb_Utilities_hh

#ifndef _GNU_SOURCE
/** Defined here, if not already defined, before `<pthread.h>` is included. */
#  define _GNU_SOURCE
#endif
#include <pthread.h>
#include <cstdint>                      // uint32_t
#include <string>

#include "rapidjson/document.h"

namespace Pds
{
  namespace Fabrics { class Fabric; }

  /** Namespace of the psdaq event-builder (eb) code. */
  namespace Eb
  {
    /** Return size rounded up to a whole number of system pages (sysconf(_SC_PAGESIZE)). */
    size_t roundUpSize(size_t size);
    /** Allocate a page-aligned block of roundUpSize(size) bytes with posix_memalign(). Returns nullptr (after printing the error to stderr) on failure; the memory is not zeroed unless built with VALGRIND defined. */
    void*  allocRegion(size_t size);

    /// Log the libfabric configuration actually negotiated, once per process.  Repeat
    /// calls are ignored, so every transport may call it without coordinating.
    void   logFabricInfo(const char* who, Fabrics::Fabric* fabric);

    /** Static helpers that pack and unpack the 32-bit immediate data word: a 2-bit flags field (bits 31-30), a 6-bit source field (bits 29-24) and a 24-bit index field (bits 23-0). */
    class ImmData
    {
    private:
      enum { v_flg = 30, k_flg =  2 };  // Modifier flags (see Flags enum below)
      enum { v_src = 24, k_src =  6 };  // Limit to 64 Ctrbs
      enum { v_idx =  0, k_idx = 24 };  // Multiplied by pulseId tick gives time range
    private:
      enum { m_flg = ((1 << k_flg) - 1), s_flg = (m_flg << v_flg) };
      enum { m_src = ((1 << k_src) - 1), s_src = (m_src << v_src) };
      enum { m_idx = ((1 << k_idx) - 1), s_idx = (m_idx << v_idx) };
    private:
      enum { v_rsp = 1, k_rsp =  1 };
      enum { v_buf = 0, k_buf =  1 };
    public:
      /** Mask and shifted mask of the response bit (bit 1) within the flags field. */
      enum { m_rsp = ((1 << k_rsp) - 1), /**< Mask of the response bit before shifting (1). */ s_rsp = (m_rsp << v_rsp)  /**< Response bit in place (value 2); used by rsp(). */ };
      /** Mask and shifted mask of the buffer bit (bit 0) within the flags field. */
      enum { m_buf = ((1 << k_buf) - 1), /**< Mask of the buffer bit before shifting (1). */ s_buf = (m_buf << v_buf)  /**< Buffer bit in place (value 1); used by buf(). */ };
    public:
      // The ImmData word must not be able to become zero for non-L1Accepts.
      // The Monitor request server protocol depends on this
      /** Values for the flags field. Bit 0 selects Transition or Buffer, bit 1 selects Response or NoResponse. The original comment notes that the word must not become zero for non-L1Accepts because the monitor request protocol depends on it. */
      enum Flags { Transition = 0 << 0, /**< Bit 0 clear (0). */ Buffer     = 1 << 0, /**< Bit 0 set (1). */ 
                   Response   = 0 << 1, /**< Bit 1 clear (0). */ NoResponse = 1 << 1  /**< Bit 1 set (2). */ };
      /** The four combinations of the Flags bits. */
      enum { Response_Transition   = Response   | Transition, /**< Response | Transition (0). */ // 0
             Response_Buffer       = Response   | Buffer, /**< Response | Buffer (1). */     // 1
             NoResponse_Transition = NoResponse | Transition, /**< NoResponse | Transition (2). */ // 2
             NoResponse_Buffer     = NoResponse | Buffer  /**< NoResponse | Buffer (3). */ };   // 3
      /** Largest values that fit in the source and index fields. */
      enum { MaxSrc = m_src, /**< Largest source value, 63 (6-bit mask). */ MaxIdx = m_idx  /**< Largest index value, 0xffffff (24-bit mask). */ };
    public:
      /** Does nothing; empty body. */
      ImmData()  { }
      /** Does nothing; empty body. */
      ~ImmData() { }
    public:
      /** Return the 2-bit flags field (bits 31-30) of data. */
      static unsigned flg(uint32_t data)             { return (data >> v_flg) & m_flg; }
      /** Return the 6-bit source field (bits 29-24) of data. */
      static unsigned src(uint32_t data)             { return (data >> v_src) & m_src; }
      /** Return the 24-bit index field (bits 23-0) of data. */
      static unsigned idx(uint32_t data)             { return (data >> v_idx) & m_idx; }
    public:
      /** Return data with its flags field replaced by v (v is masked to the field width). */
      static unsigned flg(uint32_t data, unsigned v) { return (data & ~s_flg) | ((v << v_flg) & s_flg); }
      /** Return data with its source field replaced by v (v is masked to the field width). */
      static unsigned src(uint32_t data, unsigned v) { return (data & ~s_src) | ((v << v_src) & s_src); }
      /** Return data with its index field replaced by v (v is masked to the field width). */
      static unsigned idx(uint32_t data, unsigned v) { return (data & ~s_idx) | ((v << v_idx) & s_idx); }
    public:
      /** Build an immediate data word from flags f, source s and index i, each masked to its field width. */
      static uint32_t value(unsigned f, unsigned s, unsigned i) { return ( ((f & m_flg) << v_flg) |
                                                                           ((s & m_src) << v_src) |
                                                                           ((i & m_idx) << v_idx) ); }
    public:
      /** Return the NoResponse bit of flags f (2 if set, else 0). */
      static unsigned rsp(unsigned f) { return f & s_rsp; }
      /** Return the Buffer bit of flags f (1 if set, else 0). */
      static unsigned buf(unsigned f) { return f & s_buf; }
    };
  };
};

#endif

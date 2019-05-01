#ifndef Pds_Eb_Utilities_hh
#define Pds_Eb_Utilities_hh

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <pthread.h>
#include <cstdint>                      // uint32_t
#include <string>

#include "rapidjson/document.h"


namespace Pds
{
  namespace Eb
  {
    size_t roundUpSize(size_t size);
    void*  allocRegion(size_t size);
    void   pinThread(const pthread_t& th, int cpu);
    int    fetchFromCfgDb(const std::string& section, rapidjson::Document& top);

    class ImmData
    {
    private:
      enum { v_flg = 28, k_flg =  2 };  // Modifier flags (see Flags enum below)
      enum { v_src = 16, k_src =  6 };  // Limit to 64 Ctrbs (expandable to 2048)
      enum { v_idx =  0, k_idx = 16 };  // Multiplied by batch duration gives time range
    private:
      enum { m_flg = ((1 << k_flg) - 1), s_flg = (m_flg << v_flg) };
      enum { m_src = ((1 << k_src) - 1), s_src = (m_src << v_src) };
      enum { m_idx = ((1 << k_idx) - 1), s_idx = (m_idx << v_idx) };
    private:
      enum { v_rsp = 1, k_rsp =  1 };
      enum { v_buf = 0, k_buf =  1 };
    public:
      enum { m_rsp = ((1 << k_rsp) - 1), s_rsp = (m_rsp << v_rsp) };
      enum { m_buf = ((1 << k_buf) - 1), s_buf = (m_buf << v_buf) };
    public:
      // The ImmData word must not be able to become zero for non-L1Accepts.
      // The Monitor request server protocol depends on this
      enum Flags { Transition = 0 << 0, Buffer     = 1 << 0,
                   Response   = 0 << 1, NoResponse = 1 << 1 };
      enum { MaxSrc = m_src, MaxIdx = m_idx };
    public:
      ImmData()  { }
      ~ImmData() { }
    public:
      static unsigned flg(uint32_t data)             { return (data >> v_flg) & m_flg; }
      static unsigned src(uint32_t data)             { return (data >> v_src) & m_src; }
      static unsigned idx(uint32_t data)             { return (data >> v_idx) & m_idx; }
    public:
      static unsigned flg(uint32_t data, unsigned v) { return (data & ~s_flg) | ((v << v_flg) & s_flg); }
      static unsigned src(uint32_t data, unsigned v) { return (data & ~s_src) | ((v << v_src) & s_src); }
      static unsigned idx(uint32_t data, unsigned v) { return (data & ~s_idx) | ((v << v_idx) & s_idx); }
    public:
      static uint32_t value(unsigned f, unsigned s, unsigned i) { return ( ((f & m_flg) << v_flg) |
                                                                           ((s & m_src) << v_src) |
                                                                           ((i & m_idx) << v_idx) ); }
    public:
      static unsigned rsp(unsigned f) { return f & s_rsp; }
      static unsigned buf(unsigned f) { return f & s_buf; }
    };
  };
};

#endif

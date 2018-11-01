#ifndef Pds_Eb_Utilities_hh
#define Pds_Eb_Utilities_hh

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <pthread.h>
#include <cstdint>                      // uint32_t

namespace Pds
{
  namespace Eb
  {
    size_t roundUpSize(size_t size);
    void*  allocRegion(size_t size);
    void   pinThread(const pthread_t& th, int cpu);

    class ImmData
    {
    private:
      enum { v_spc = 30, k_spc =  2 };
      enum { v_src = 24, k_src =  6 };
      enum { v_idx =  0, k_idx = 24 };
    private:
      enum { m_spc = ((1 << k_spc) - 1), s_spc = (m_spc << v_spc) };
      enum { m_src = ((1 << k_src) - 1), s_src = (m_src << v_src) };
      enum { m_idx = ((1 << k_idx) - 1), s_idx = (m_idx << v_idx) };
    public:
      enum Space { Reserved_0, Buffer, Transition, Reserved_3 };
    public:
      ImmData()  { }
      ~ImmData() { }
    public:
      static unsigned spc(uint32_t data)             { return (data >> v_spc) & m_spc; }
      static unsigned src(uint32_t data)             { return (data >> v_src) & m_src; }
      static unsigned idx(uint32_t data)             { return (data >> v_idx) & m_idx; }
    public:
      static unsigned spc(uint32_t data, unsigned v) { return (data & ~s_spc) | ((v << v_spc) & s_spc); }
      static unsigned src(uint32_t data, unsigned v) { return (data & ~s_src) | ((v << v_src) & s_src); }
      static unsigned idx(uint32_t data, unsigned v) { return (data & ~s_idx) | ((v << v_idx) & s_idx); }
    public:
      static uint32_t transition(unsigned s, unsigned i) { return ( ((Transition & m_spc) << v_spc) |
                                                                    ((s          & m_src) << v_src) |
                                                                    ((i          & m_idx) << v_idx) ); }
      static uint32_t buffer    (unsigned s, unsigned i) { return ( ((Buffer     & m_spc) << v_spc) |
                                                                    ((s          & m_src) << v_src) |
                                                                    ((i          & m_idx) << v_idx) ); }
    };
  };
};

#endif

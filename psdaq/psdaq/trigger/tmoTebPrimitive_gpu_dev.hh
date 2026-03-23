#pragma once

#include <cstddef>
#include <cstdint>

#include "drpGpu/MemPool.hh"            // For DmaDsc
#include "psdaq/service/EbDgram.hh"     // For TimingHeader

namespace Pds {
  namespace Trg {

    struct TmoTebEventFn
    {
      __device__
      void operator()(float     const* const /*calibBuffers*/,
                      size_t    const        /*calibBufsCnt*/,
                      uint32_t* const* const out,
                      size_t    const        outBufsCnt,
                      unsigned  const        index,
                      unsigned  const        /*nPanels*/) const
      {
        // Analyze calibBuffers for all panels to determine TEB input data for the trigger
        //float* __restrict__ calibBuf = &calibBuffers[index * calibBufsCnt]; // nPanels * nElements of data follow

        constexpr uint32_t write_{0xdeadbeef};
        constexpr uint32_t monitor_{0x12345678};

        // Although this also runs for transitions, it is harmless, but it could be tested for here
        constexpr unsigned tebInpOs = (sizeof(Drp::Gpu::DmaDsc) + sizeof(Pds::TimingHeader)) / sizeof(**out);
        uint32_t* const    tebInp   = &out[0][index * outBufsCnt + tebInpOs];   // Only panel 0 receives the summary
        tebInp[0] = write_;
        tebInp[1] = monitor_;
      }
    };

  } // Trg
} // Pds

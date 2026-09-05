#include "EpixUHR3x2.hh"

#include "ReaderKernels.cuh"

#include <cuda_fp16.h>

#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

struct uhr3x2_domain{ static constexpr char const* name{"EpixUHR3x2"}; };
using uhr3x2_scoped_range = nvtx3::scoped_range_in<uhr3x2_domain>;


namespace Drp {
  class PGPEvent;
  namespace Gpu {

class RawDef : public VarDef
{
public:
  enum index
    {
      raw
    };

  RawDef()
  {
    Alg raw("raw", 0, 0, 0);
    NameVec.push_back({"raw", Name::UINT8, 1});
  }
};
  } // Gpu
} // Drp

EpixUHR3x2::EpixUHR3x2(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Use the CPU-side Drp::EpixUHR3x2 for panel setup, configuration and the
  // Python-driven bits.  The data description and handling in the GPU case are
  // different, so only that portion is borrowed.
  _initialize<Drp::EpixUHR3x2>(para, pool);

  // Check there is enough space in the DMA buffers for this many pixels.  The
  // AxiStream Batcher adds a header line plus a tail line per sub-frame, and
  // pads each sub-frame up to a line boundary, so allow for that.  Not an
  // assert(): the GPU DRP is built with NDEBUG, which would compile it out.
  constexpr size_t maxLineWidth{64};     // Widest AXI stream this can arrive on
  constexpr size_t batchOverhead{(1 + NumSubFrames) * maxLineWidth};
  constexpr size_t minDmaSize{NPixels * sizeof(__half) + sizeof(TimingHeader) + batchOverhead};
  if (minDmaSize > pool.dmaSize()) {
    logging::critical("DMA buffer of %zu bytes is too small for %u pixels in %u sub-frames: need %zu",
                      pool.dmaSize(), NPixels, NumSubFrames, minDmaSize);
    abort();
  }

  // Set up buffers
  pool.createCalibBuffers(NPixels);
}

EpixUHR3x2::~EpixUHR3x2()
{
  auto pool = m_pool->getAs<MemPoolGpu>();
  pool->destroyCalibBuffers();
}

unsigned EpixUHR3x2::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::EpixUHR3x2 configure");

  // Configure the CPU-side detector for the panel
  unsigned rc = m_det->configure(config_alias, xtc, bufEnd);
  if (rc) {
    logging::error("Gpu::EpixUHR3x2::configure failed for %s\n", m_para->device);
    return rc;
  }

  Alg alg("raw", 0, 0, 0);
  NamesId namesId(nodeId, EventNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para->detName.c_str(), alg,
                                         m_para->detType.c_str(), m_para->serNo.c_str(), namesId, m_para->detSegment);
  RawDef dataDef;
  names.add(xtc, bufEnd, dataDef);
  m_namesLookup[namesId] = NameIndex(names);

  logging::info("Gpu::EpixUHR3x2 configure: xtc size %u", xtc.sizeofPayload());

  return 0;
}

unsigned EpixUHR3x2::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  unsigned rc = m_det->beginrun(xtc, bufEnd, runInfo);
  if (rc) {
    logging::error("Gpu::EpixUHR3x2::beginrun failed for %s\n", m_para->device);
    return rc;
  }

  // Nothing to upload: the panel's data is calibrated in the detector's firmware
  return rc;
}

void EpixUHR3x2::event(Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count)
{
  constexpr uint32_t lane{0}; // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  size_t size = buffer->size;

  // The batched payload is at least the pixel data plus the TimingHeader; the
  // batcher's own header, tails and line padding make it somewhat larger
  constexpr auto minEventSize{sizeof(TimingHeader) + NPixels * sizeof(__half)};
  if      (size  < minEventSize)       dgram.xtc.damage.increase(Damage::MissingData);
  else if (size == m_pool->dmaSize())  dgram.xtc.damage.increase(Damage::Truncated);

  // @todo: Deal with prescaled raw for the panel here?
}

// The panel's data is calibrated fp16 from firmware, so the per-element work is
// a width conversion.  The policy also owns the placement of each ASIC's
// sub-frame within the calibrated buffer, because sub-frame tdests are not
// necessarily a contiguous run and only the Detector knows the mapping.
struct EpixUHR3x2Calib
{
  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.batched)  return;            // A transition: payload is a TimingHeader
    // A failed scan means the payload is unintelligible, so don't interpret it.
    // _waitForDMA has reported it and the host sees the latched status.
    if (!pyld.subFrames->ok())  return;

    auto const strideCnt = pyld.outCnt / EpixUHR3x2::NumAsics;
    for (unsigned k = 0; k < EpixUHR3x2::NumAsics; ++k) {
      auto const& sub = (*pyld.subFrames)[EpixUHR3x2::FirstDataTdest + k];
      auto const  off = k * strideCnt;
      auto const  cnt = sub.size / sizeof(__half);
      if (cnt == 0) {                   // Withheld ASIC: clear the hole it leaves
        for (auto i = tid; i < strideCnt; i += stride)  pyld.out[off + i] = 0.f;
        continue;
      }
      auto const __restrict__ src = (__half const*)sub.data(pyld.data);
      auto const              nElem = cnt > strideCnt ? strideCnt : cnt;
      for (auto i = tid; i < nElem; i += stride)  pyld.out[off + i] = __half2float(src[i]);
    }
  }
};

// Instantiating the kernel template here puts the conversion in the same CUDA
// module as the kernel, so it inlines.  See ReaderKernels.cuh.
void EpixUHR3x2::recordEvent(cudaStream_t           stream,
                             unsigned               blocks,
                             unsigned               threads,
                             const EventKernelArgs& args)
{
  _event<EpixUHR3x2Calib><<<blocks, threads, 0, stream>>>(args, EpixUHR3x2Calib{});
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHR3x2(para, pool);
}

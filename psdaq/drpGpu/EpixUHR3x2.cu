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

// The pass-through description, for when the panel's data is recorded as it
// arrives rather than calibrated and reduced.  Unlike RawDef above -- which
// describes a Reducer's output, an opaque blob whose interpretation the Xtc
// header carries -- this describes real u16 pixels, so it is typed and shaped to
// match Drp::EpixUHR3x2RawDef in the CPU DRP (drp/EpixUHR3x2.cc).  Offline has to
// see the same array whichever DRP wrote it.
class RawU16Def : public VarDef
{
public:
  enum index
    {
      raw
    };

  RawU16Def()
  {
    NameVec.push_back({"raw", Name::UINT16, 2});
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

  // Pass-through mode: record the panel's data as it arrives, uncalibrated and
  // unreduced, instead of converting it and handing it to a Reducer.  This is
  // what allows the GPU DRP to run against the hardware emulator, which produces
  // u16 (1 gain bit, 11-bit ADC value, 4 zero bits) rather than the fp16 the
  // firmware will eventually produce.
  // @todo: Stage 2 replaces this kwarg with a test of the CALIB config alias,
  //        which PGPDrp::configure() has in its msg.
  if (para.kwargs.find("raw") != para.kwargs.end())
    m_passthru = std::stoul(para.kwargs.at("raw")) != 0;
  if (m_passthru)
    logging::warning("EpixUHR3x2: pass-through mode -- recording raw u16, "
                     "uncalibrated and unreduced");

  // Check there is enough space in the DMA buffers for this many pixels.  The
  // AxiStream Batcher adds a header line plus a tail line per sub-frame, and
  // pads each sub-frame up to a line boundary, so allow for that.  The line
  // size is in the header's 4-bit width field.  It is not necessarily the
  // same as the PCIe frame size.  Not an assert(): the GPU DRP is built with
  // NDEBUG, which would compile it out.
  // sizeof(__half) covers pass-through too: u16 pixels are the same 2 bytes.
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
  // In pass-through mode this Detector's own description is what offline reads,
  // because no Reducer runs to supply one, so it must describe the real u16
  // pixels.  Otherwise the recorded payload is the Reducer's and this describes
  // only the untyped byte blob.
  if (m_passthru) {
    RawU16Def dataDef;
    names.add(xtc, bufEnd, dataDef);
  } else {
    RawDef dataDef;
    names.add(xtc, bufEnd, dataDef);
  }
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

// Normal running, i.e. the BEAM config alias: the panel's data is calibrated fp16
// from firmware, so the per-element work is a width conversion.  The policy also
// owns the placement of each ASIC's sub-frame within the calibrated buffer,
// because sub-frame tdests are not necessarily a contiguous run and only the
// Detector knows the mapping.
//
// Named for the alias rather than for what it produces: "Calib" would be
// ambiguous, since the CALIB alias selects the mode that records *un*calibrated
// data.  See EpixUHR3x2Calib below.
struct EpixUHR3x2Beam
{
  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    // hasData is false for a transition -- whose batch has no data sub-frames --
    // and for any payload whose size or layout the Reader did not recognise
    if (!pyld.hasData)  return;

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

// Calibration running, i.e. the CALIB config alias, and for now also the
// pass-through mode that lets this Detector run against the u16 hardware
// emulator.  The panel's data is copied to the raw block exactly as it arrives:
// no pedestal or gain applied, no width conversion, and no Reducer afterwards.
//
// It writes pyld.raw, not pyld.out: the calibrated buffer is bypassed entirely,
// so nothing populates it in this mode.  Anything downstream that reads it --
// a TrgInpGen, say -- would be reading a stale or uninitialised buffer, which is
// why the trigger input must be data-independent here and always vote to persist
// and monitor.
struct EpixUHR3x2Calib
{
  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.hasData)  return;         // A transition, or nothing intelligible
    if (!pyld.raw)      return;         // No raw block: misconfigured, not our call

    // The array offline sees is a fixed [NumAsics][AsicPixels], as the CPU DRP
    // writes, so a missing or short ASIC leaves zeros in its region rather than
    // shrinking the array.  Damage::MissingData in event() carries the fact.
    auto const __restrict__ dst = (uint16_t*)pyld.raw;
    auto const rawCnt   = pyld.rawCnt / sizeof(uint16_t);
    auto const strideCnt = rawCnt / EpixUHR3x2::NumAsics;
    for (unsigned k = 0; k < EpixUHR3x2::NumAsics; ++k) {
      auto const& sub = (*pyld.subFrames)[EpixUHR3x2::FirstDataTdest + k];
      auto const  off = k * strideCnt;
      auto const  cnt = sub.size / sizeof(uint16_t);
      auto const  nElem = cnt > strideCnt ? strideCnt : cnt;
      auto const __restrict__ src = (uint16_t const*)sub.data(pyld.data);
      // One pass over the whole ASIC region: copy what arrived, zero the rest.
      // A withheld ASIC has nElem == 0 and so is zeroed entirely.  Branchless
      // in the common case where nElem == strideCnt.
      for (auto i = tid; i < strideCnt; i += stride)
        dst[off + i] = i < nElem ? src[i] : 0;
    }
  }
};

// Instantiating the kernel templates here puts the per-element work in the same
// CUDA module as the kernel, so it inlines.  See ReaderKernels.cuh.
void EpixUHR3x2::recordEvent(cudaStream_t           stream,
                             unsigned               blocks,
                             unsigned               threads,
                             const EventKernelArgs& args)
{
  if (m_passthru)
    _event<EpixUHR3x2Calib><<<blocks, threads, 0, stream>>>(args, EpixUHR3x2Calib{});
  else
    _event<EpixUHR3x2Beam ><<<blocks, threads, 0, stream>>>(args, EpixUHR3x2Beam {});
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHR3x2(para, pool);
}

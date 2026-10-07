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
  } // Gpu
} // Drp

EpixUHR3x2::EpixUHR3x2(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Use the CPU-side Drp::EpixUHR3x2 for panel setup, configuration and the
  // Python-driven bits.  The data description and handling in the GPU case are
  // different, so only that portion is borrowed.
  _initialize<Drp::EpixUHR3x2>(para, pool);

  // What format the panel's data arrives in, which is a property of its firmware:
  //
  //   raw=u16    1 gain bit in bit 0, an 11-bit ADC value in bits 1-11, zeros in
  //              bits 12-15, so the GPU applies pedestals and gains itself.  What
  //              the firmware presents today, and so the default.
  //   raw=fp16   already calibrated by the firmware; the per-element work is an
  //              fp16 -> fp32 conversion.  REFUSED -- see below.
  //
  // Either way a Reducer runs on the calibrated result.
  //
  // fp16 is deliberately unselectable rather than deleted.  The planned firmware picks
  // u16 or fp16 by register, so the raw block is one or the other and the non-raw block
  // becomes signal-extracted "fex" data.  The per-element work for fp16 and the fex
  // format are both unsettled, so EpixUHR3x2Beam's conversion cannot be completed.  It
  // stays compiled -- recordEvent() still instantiates it -- so it keeps building as
  // the surrounding code changes.  See TODO.md.
  //
  // @todo: Read the format from the firmware rather than from a kwarg.  It is
  //        queryable -- the Build String and Firmware Version in /proc/datadev_* are
  //        available at Configure, and BEBDetector already reads that region -- so
  //        this could be derived instead of asserted, and a mismatch between what the
  //        firmware sends and what the kwarg claims would stop being silent.
  if (para.kwargs.find("raw") != para.kwargs.end()) {
    auto const& fmt = para.kwargs.at("raw");
    if      (fmt == "u16")   m_u16 = true;
    else if (fmt == "fp16") {
      logging::critical("EpixUHR3x2: 'raw=fp16' is not supported yet.  It awaits the "
                        "firmware's mode registers and the fex format, and the Xtc type "
                        "system has no fp16 to describe it with; "
                        "use 'raw=u16', which is what the firmware presents today.");
      abort();
    }
    else {
      logging::critical("EpixUHR3x2: unrecognized 'raw=%s'.  Expected 'u16' (the "
                        "default).", fmt.c_str());
      abort();
    }
  }

  // Nb: whether to record raw rather than reduced is not a kwarg: the CALIB config
  // alias decides it, per Configure.  See configure() and Gpu::Detector::setPassthru().
  logging::warning("EpixUHR3x2: data format %s%s", m_u16 ? "u16" : "fp16",
                   m_u16 ? " -- gain bit 0, ADC bits 1-11, calibrated on the GPU"
                         : " -- calibrated by the firmware");

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

  // Space for the calibration constants, one plane per gain range.  Only u16 mode
  // applies them; the fp16 path arrives calibrated from firmware.
  if (m_u16) {
    chkError(cudaMalloc(&m_peds_d,  NRanges * NPixels * sizeof(*m_peds_d)));
    chkError(cudaMalloc(&m_gains_d, NRanges * NPixels * sizeof(*m_gains_d)));
  }
}

EpixUHR3x2::~EpixUHR3x2()
{
  auto pool = m_pool->getAs<MemPoolGpu>();
  if (m_gains_d)  chkError(cudaFree(m_gains_d));
  if (m_peds_d)   chkError(cudaFree(m_peds_d));
  pool->destroyCalibBuffers();
}

unsigned EpixUHR3x2::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::EpixUHR3x2 configure: alias '%s'%s", config_alias.c_str(),
                m_passthru ? ", recording raw u16 uncalibrated and unreduced" : "");

  // Configure the CPU-side detector for the panel
  unsigned rc = m_det->configure(config_alias, xtc, bufEnd);
  if (rc) {
    logging::error("Gpu::EpixUHR3x2::configure failed for %s\n", m_para->device);
    return rc;
  }

  // Drp::EpixUHR3x2 has already declared the panel's event Names under this same
  // NamesId, as the typed and shaped u16 array this code needs: the raw block is u16
  // in both modes, the whole payload in pass-through and the Reducer's companion in
  // BEAM.  Declaring a second Names here is what psana's NamesIter rejects as a
  // duplicate namesId, so adopt the base class's NameIndex instead -- rawEvent()
  // needs the entry in this Detector's lookup for CreateData, not another block in
  // the Xtc.  NameIndex assignment deep-copies, so it outlives m_det either way.
  NamesId namesId(nodeId, EventNamesIndex);
  auto& baseLookup = m_det->namesLookup();
  if (baseLookup.find(namesId) == baseLookup.end()) {
    logging::error("Gpu::EpixUHR3x2::configure: Drp::EpixUHR3x2 declared no Names "
                   "for namesId 0x%x", unsigned(namesId));
    return 1;
  }
  m_namesLookup[namesId] = baseLookup[namesId];

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

  // In u16 mode the GPU applies pedestals and gains, so they have to be uploaded.
  // With the fp16 firmware there is nothing to upload: the panel's data arrives
  // already calibrated.
  if (m_u16) {
    // @todo: Fetch calibration constants.  Fabricating them -- pedestal 0, gain 1 --
    //        makes the calibrated values numerically equal to the raw ADC counts, so
    //        this proves the path but not the science.  See the "Fetch calibration
    //        constants" item in TODO.md; EpixUHRemu and Jungfrau do the same.
    std::vector<float> peds (NPixels, 0.0);
    std::vector<float> gains(NPixels, 1.0);
    auto peds_d  = m_peds_d;
    auto gains_d = m_gains_d;
    for (unsigned range = 0; range < NRanges; ++range) {
      chkError(cudaMemcpy(peds_d,  peds.data(),  NPixels * sizeof(*peds_d),  cudaMemcpyDefault));
      chkError(cudaMemcpy(gains_d, gains.data(), NPixels * sizeof(*gains_d), cudaMemcpyDefault));
      peds_d  += NPixels;
      gains_d += NPixels;
    }
  }

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

// Copies the panel's u16 pixels into the raw block, uncalibrated, walking the ASIC
// sub-frames exactly as the calibrating policies do.  Shared by CALIB mode, where the
// raw block is the whole payload, and by prescaling, where it accompanies the
// calibrated data.
//
// pyld.raw being set is what says this event has a raw block to fill, in either mode:
// the Reader sets it per event, for all of them in pass-through and for marked ones
// when prescaling.  A policy must not test keepRaw itself -- in pass-through that bit
// is set on ~1 Hz of events whose payload is *entirely* raw, so honouring it would
// leave the rest of the frames zeroed.
static __device__
void _copyRaw(const EventPayload& pyld, unsigned tid, unsigned stride)
{
  if (!pyld.raw)  return;               // No raw block this event: nothing to fill

  // The array offline sees is a fixed [NumAsics][AsicPixels], as the CPU DRP writes,
  // so a missing or short ASIC leaves zeros in its region rather than shrinking the
  // array.  Damage::MissingData in event() carries the fact.
  auto const __restrict__ dst = (uint16_t*)pyld.raw;
  auto const rawCnt    = pyld.rawCnt / sizeof(*dst);
  auto const strideCnt = rawCnt / EpixUHR3x2::NumAsics;
  for (unsigned k = 0; k < EpixUHR3x2::NumAsics; ++k) {
    auto const& sub = (*pyld.subFrames)[EpixUHR3x2::FirstDataTdest + k];
    auto const  off = k * strideCnt;
    auto const __restrict__ src = (uint16_t const*)sub.data(pyld.data);
    auto const  cnt = sub.size / sizeof(*src);
    auto const  nElem = cnt > strideCnt ? strideCnt : cnt;
    // One pass over the whole ASIC region: copy what arrived, zero the rest.  A
    // withheld ASIC has nElem == 0 and so is zeroed entirely.  Branchless in the
    // common case where nElem == strideCnt.
    for (auto i = tid; i < strideCnt; i += stride)
      dst[off + i] = i < nElem ? src[i] : 0;
  }
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

// The u16 payload's policy: calibrate each pixel on the GPU into the calibrated
// buffer, where a Reducer then finds it, rather than converting fp16.
//
// Same sub-frame walk as EpixUHR3x2Beam, so the two differ only in the per-element
// work.  The gain bit selects which pedestal/gain plane applies, exactly as it does
// for the detectors whose range bits sit above the data -- pedGainCalibrate() is told
// where both fields are rather than assuming an order.
struct EpixUHR3x2U16
{
  float const* peds;
  float const* gains;
  unsigned     rangeOffset;
  unsigned     rangeBits;
  unsigned     dataOffset;
  unsigned     dataBits;

  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.hasData)  return;         // A transition, or nothing intelligible

    auto const strideCnt = pyld.outCnt / EpixUHR3x2::NumAsics;
    for (unsigned k = 0; k < EpixUHR3x2::NumAsics; ++k) {
      auto const& sub = (*pyld.subFrames)[EpixUHR3x2::FirstDataTdest + k];
      auto const  off = k * strideCnt;
      auto const __restrict__ src = (uint16_t const*)sub.data(pyld.data);
      auto const  cnt = sub.size / sizeof(*src);
      if (cnt == 0) {                   // Withheld ASIC: clear the hole it leaves
        for (auto i = tid; i < strideCnt; i += stride)  pyld.out[off + i] = 0.f;
        continue;
      }
      auto const  nElem = cnt > strideCnt ? strideCnt : cnt;
      // pgOffset places this ASIC's pixels within the pedestal/gain plane, whose
      // stride is the whole frame
      pedGainCalibrate(&pyld.out[off], src, nElem, rangeOffset, rangeBits,
                       dataOffset, dataBits,
                       peds, gains, pyld.outCnt, off, nullptr, tid, stride);
      // Clear the tail a short ASIC leaves, so a partial frame does not show stale
      // data from a previous event.  Every thread strides over the whole region and
      // skips what was just calibrated, rather than starting at tid + nElem, which
      // would leave holes.
      for (auto i = tid; i < strideCnt; i += stride)
        if (i >= nElem)  pyld.out[off + i] = 0.f;
    }
  }
};

// Calibration running, i.e. what the CALIB config alias selects, and also what the
// transitional `raw=1` selects.  The panel's data is copied to the raw block exactly as it arrives:
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
    _copyRaw(pyld, tid, stride);
  }
};

// BEAM mode with the u16 firmware: calibrate every event, and on the ~1 Hz the timing
// system marked, copy the uncalibrated u16 as well so offline can reproduce the
// calibration from it.  Both read the same sub-frames; the Reader gives a marked event
// a raw block and the rest none, so _copyRaw's own test is what selects them.
struct EpixUHR3x2Prescale
{
  EpixUHR3x2U16 calib;

  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    calib.process(pyld, tid, stride);
    if (pyld.hasData)  _copyRaw(pyld, tid, stride);
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
  else if (m_u16) {
    EpixUHR3x2U16 const u16{pedestals_d(), gains_d(),
                            rangeOffset(), rangeBits(),
                            dataOffset(),  dataBits()};
    // Prescaling rides along with the calibration: _copyRaw fills the raw block on
    // the events the Reader gave one to, and does nothing on the rest
    _event<EpixUHR3x2Prescale><<<blocks, threads, 0, stream>>>(args, EpixUHR3x2Prescale{u16});
  }
  else
    // Unreachable: the ctor refuses raw=fp16.  Instantiated so that this policy keeps
    // compiling until the combined u16+fp16 firmware layout is settled.
    _event<EpixUHR3x2Beam ><<<blocks, threads, 0, stream>>>(args, EpixUHR3x2Beam {});
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHR3x2(para, pool);
}

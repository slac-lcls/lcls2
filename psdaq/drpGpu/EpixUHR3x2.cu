#include "EpixUHR3x2.hh"
#include "EpixUHRcalib.hh"

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

  // The f/w sends the panel as one 336x576 image, so psana must not reassemble it
  static_cast<Drp::EpixUHR3x2*>(m_det)->setRawAlgVersion(1, 0, 0);

  // The data format, which depends on the firmware:
  //   raw=u16    gain bit 0, ADC bits 1-11; calibrated on the GPU.  The default.
  //   raw=fp16   calibrated by the firmware.  Refused: EpixUHR3x2Beam is incomplete,
  //              but recordEvent() still instantiates it so it keeps compiling.
  // @todo: Derive the format from the firmware's Build String in /proc/datadev_*,
  //        so a wrong kwarg stops being silent.
  if (para.kwargs.find("raw") != para.kwargs.end()) {
    auto const& fmt = para.kwargs.at("raw");
    if      (fmt == "u16")   m_u16 = true;
    else if (fmt == "fp16") {
      logging::critical("EpixUHR3x2: 'raw=fp16' is not supported; use 'raw=u16'.");
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

  // Check the DMA buffers fit the pixels plus the batcher's header line and a tail
  // line per sub-frame.  Not an assert(): the GPU DRP is built with NDEBUG.
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

  // Adopt the u16 Names Drp::EpixUHR3x2 already declared under this NamesId: psana
  // rejects a second declaration as a duplicate namesId.  The copy is deep.
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

  // Upload constants only in u16 BEAM: pass-through records raw, so a CALIB run can
  // take darks for a panel with no constants yet.
  if (m_u16 && !m_passthru) {
    // The panel's serial, which Drp::EpixUHR3x2 read from it at Connect.
    std::vector<float> peds (NRanges * NPixels, 0.0);
    std::vector<float> gains(NRanges * NPixels, 1.0);
    if (m_para->serNo.empty()) {
      logging::error("Gpu::EpixUHR3x2: no serial number to fetch pedestals for");
      rc = 1;
    }
    else {
      if (fetchEpixUHRconstants("pedestals", m_para->serNo, runInfo, EpixUHRgainModes, NRanges, peds, m_warnings))
        rc = 1;
      if (fetchEpixUHRconstants("gain_factors", m_para->serNo, runInfo, EpixUHRgainModes, NRanges, gains, m_warnings))
        rc = 1;
    }
    if (rc)  m_error = "failed to fetch calibration constants";
    chkError(cudaMemcpy(m_peds_d,  peds.data(),  peds.size()  * sizeof(*m_peds_d),  cudaMemcpyDefault));
    chkError(cudaMemcpy(m_gains_d, gains.data(), gains.size() * sizeof(*m_gains_d), cudaMemcpyDefault));
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

// Copies the panel's u16 image into the raw block, uncalibrated, for CALIB and for
// prescaling.  Test pyld.raw, never keepRaw: in pass-through honouring keepRaw would
// zero every unmarked frame.
static __device__
void _copyRaw(const EventPayload& pyld, unsigned tid, unsigned stride)
{
  if (!pyld.raw)  return;               // No raw block this event: nothing to fill

  // A short image leaves zeros in the fixed [NumRows][NumCols]; MissingData says so
  auto const& sub = (*pyld.subFrames)[EpixUHR3x2::DataTdest];
  auto const __restrict__ src = (uint16_t const*)sub.data(pyld.data);
  auto const __restrict__ dst = (uint16_t*)pyld.raw;
  auto const rawCnt = pyld.rawCnt / sizeof(*dst);
  auto const cnt    = sub.size / sizeof(*src);
  auto const nElem  = cnt > rawCnt ? rawCnt : cnt;
  for (auto i = tid; i < rawCnt; i += stride)
    dst[i] = i < nElem ? src[i] : 0;
}

// BEAM with fp16 firmware: the per-element work is a width conversion.  Named for the
// alias, since the CALIB alias records *un*calibrated data.
struct EpixUHR3x2Beam
{
  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.hasData)  return;         // A transition, or nothing intelligible

    auto const& sub = (*pyld.subFrames)[EpixUHR3x2::DataTdest];
    auto const __restrict__ src = (__half const*)sub.data(pyld.data);
    auto const cnt   = sub.size / sizeof(*src);
    auto const nElem = cnt > pyld.outCnt ? pyld.outCnt : cnt;
    for (auto i = tid; i < pyld.outCnt; i += stride)
      pyld.out[i] = i < nElem ? __half2float(src[i]) : 0.f;
  }
};

// BEAM with u16 firmware: calibrate each pixel, the gain bit picking the plane
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

    auto const& sub = (*pyld.subFrames)[EpixUHR3x2::DataTdest];
    auto const __restrict__ src = (uint16_t const*)sub.data(pyld.data);
    auto const cnt   = sub.size / sizeof(*src);
    auto const nElem = cnt > pyld.outCnt ? pyld.outCnt : cnt;
    pedGainCalibrate(pyld.out, src, nElem, rangeOffset, rangeBits,
                     dataOffset, dataBits,
                     peds, gains, pyld.outCnt, 0, nullptr, tid, stride);
    // Clear the tail of a short image, so it does not show a previous event's data
    for (auto i = nElem + tid; i < pyld.outCnt; i += stride)  pyld.out[i] = 0.f;
  }
};

// CALIB (and the transitional `raw=1`): copy the data to the raw block as it arrives,
// with no Reducer after.  The calibrated buffer is never written, so the trigger input
// must be data-independent and always vote to persist and monitor.
struct EpixUHR3x2Calib
{
  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.hasData)  return;         // A transition, or nothing intelligible
    _copyRaw(pyld, tid, stride);
  }
};

// BEAM with u16 firmware and prescaling: calibrate every event, and also copy the raw
// u16 of the events the timing system marked (_copyRaw selects them).
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

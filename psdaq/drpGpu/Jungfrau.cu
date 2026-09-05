#include "Jungfrau.hh"

#include "ReaderKernels.cuh"

#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

#include <cstddef>                      // For offsetof

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

struct jf_domain{ static constexpr char const* name{"Jungfrau"}; };
using jf_scoped_range = nvtx3::scoped_range_in<jf_domain>;


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

Jungfrau::Jungfrau(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool),
  m_nModules       (0),
  m_pedsVec_d      (nullptr),
  m_gainsVec_d     (nullptr)
{
  // Count modules the same way Drp::Jungfrau's constructor does, so that the two
  // agree about which lane is which module
  for (size_t i = 0; i < PGP_MAX_LANES - 1; ++i) {
    if (para.laneMask & (1 << i))  ++m_nModules;
  }
  if (m_nModules == 0) {
    logging::critical("Jungfrau needs at least one lane in the lane mask (0x%x)", para.laneMask);
    abort();
  }

  // Drp::Jungfrau sets m_debatch for this timebase, which wraps the payload in a
  // further AxiStream Batcher layer.  The Reader's sub-frame scan handles one
  // outer batch, so refuse the configuration rather than misread the payload.
  // @todo: Support this if the GPU firmware can produce it
  auto tb = para.kwargs.find("timebase");
  if ((tb != para.kwargs.end()) && (tb->second == std::string("119M"))) {
    logging::critical("Gpu::Jungfrau does not support the 119M timebase, which "
                      "adds an outer AxiStream Batcher layer");
    abort();
  }

  // Use the CPU-side Drp::Jungfrau for panel setup, configuration and the
  // Python- and SLS-driven bits
  _initialize<Drp::Jungfrau>(para, pool);

  logging::info("Gpu::Jungfrau: %u module(s), %u pixels", m_nModules, nPixels());

  // Check there is enough space in the DMA buffers for the batched payload.  Each
  // of the two batcher levels adds a header line, and a tail line plus padding to
  // a line boundary per sub-frame.
  constexpr size_t maxLineWidth{64};    // Widest AXI stream this can arrive on
  size_t const outerSubFrames{subframeCount()};
  size_t const innerSubFrames{size_t(m_nModules) * PacketNum};
  size_t const batchOverhead{(1 + outerSubFrames + m_nModules + innerSubFrames) * maxLineWidth};
  size_t const minDmaSize{size_t(m_nModules) * PacketNum * Drp::JungfrauData::PacketSize +
                          sizeof(TimingHeader) + batchOverhead};
  if (minDmaSize > pool.dmaSize()) {
    logging::critical("DMA buffer of %zu bytes is too small for %u module(s) of "
                      "%u packets: need %zu",
                      pool.dmaSize(), m_nModules, PacketNum, minDmaSize);
    abort();
  }

  // Set up buffers
  pool.createCalibBuffers(nPixels());

  // Allocate space for the calibration constants
  chkError(cudaMalloc(&m_pedsVec_d,  NRanges * nPixels() * sizeof(*m_pedsVec_d)));
  chkError(cudaMalloc(&m_gainsVec_d, NRanges * nPixels() * sizeof(*m_gainsVec_d)));
}

Jungfrau::~Jungfrau()
{
  auto pool = m_pool->getAs<MemPoolGpu>();
  if (m_gainsVec_d)  chkError(cudaFree(m_gainsVec_d));
  if (m_pedsVec_d)   chkError(cudaFree(m_pedsVec_d));

  pool->destroyCalibBuffers();
}

unsigned Jungfrau::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::Jungfrau configure");

  // Configure the CPU-side detector for the modules
  unsigned rc = m_det->configure(config_alias, xtc, bufEnd);
  if (rc) {
    logging::error("Gpu::Jungfrau::configure failed for %s\n", m_para->device);
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

  logging::info("Gpu::Jungfrau configure: xtc size %u", xtc.sizeofPayload());

  return 0;
}

unsigned Jungfrau::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  unsigned rc = m_det->beginrun(xtc, bufEnd, runInfo);
  if (rc) {
    logging::error("Gpu::Jungfrau::beginrun failed for %s\n", m_para->device);
    return rc;
  }

  // Load the calibration constants onto the GPU
  // @todo: Fetch calibration constants.  The CPU-side Jungfrau writes raw data
  //        and leaves calibration to analysis, so there is no source for these
  //        yet; a unit pedestal and gain make the conversion a pass-through.
  auto const nPix = nPixels();
  std::vector<float> peds(nPix, 0.0);
  std::vector<float> gains(nPix, 1.0);
  auto peds_d  = m_pedsVec_d;
  auto gains_d = m_gainsVec_d;
  for (unsigned range = 0; range < NRanges; ++range) {
    chkError(cudaMemcpy(peds_d,  peds.data(),  nPix * sizeof(*peds_d),  cudaMemcpyDefault));
    chkError(cudaMemcpy(gains_d, gains.data(), nPix * sizeof(*gains_d), cudaMemcpyDefault));
    peds_d  += nPix;
    gains_d += nPix;
  }

  return rc;
}

void Jungfrau::event(Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count)
{
  constexpr uint32_t lane{0}; // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  size_t size = buffer->size;

  auto const minEventSize{sizeof(TimingHeader) +
                          size_t(m_nModules) * PacketNum * Drp::JungfrauData::PacketSize};
  if      (size  < minEventSize)       dgram.xtc.damage.increase(Damage::MissingData);
  else if (size == m_pool->dmaSize())  dgram.xtc.damage.increase(Damage::Truncated);

  // @todo: Deal with prescaled raw for the modules here?
}

// The Jungfrau's per-event work: walk the second batcher level to find each
// module's packets, scatter their pixels to where the packet headers say they
// belong, and calibrate them from the pedestals and gains.
//
// The nested packet offsets are computed rather than walked.  A serial
// EvtBatcherIterator walk of 128 packets per module would cost more than the rest
// of the kernel, and it is unnecessary: the batcher's layout is
//
//   [header lw][sub-frame 0 padded to lw][tail lw][sub-frame 1 padded][tail lw]...
//
// so with a fixed packet size the offsets follow arithmetically from the nested
// header's line width.  Each packet's tail is still read, to check its size.
struct JungfrauCalib
{
  float const* peds;
  float const* gains;
  unsigned     nModules;
  unsigned     nPixels;                 // nModules * ModulePixels, the p/g stride

  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.batched)  return;         // A transition: payload is a TimingHeader
    if (!pyld.subFrames->ok())  return; // Corrupt batch: already reported

    // Which input packet holds which output packet slot.  Rebuilt per module;
    // NotPresent marks a slot no packet claimed, whose pixels are zeroed.
    constexpr uint8_t NotPresent{0xff};
    __shared__ uint8_t slotToPacket[Jungfrau::PacketNum];

    constexpr auto pixPerPkt = Jungfrau::PixelPerPacket;
    constexpr auto pktNumOff = offsetof(Drp::JungfrauData::Header, packetnum);

    for (unsigned module = 0; module < nModules; ++module) {
      auto const& modSub = (*pyld.subFrames)[Jungfrau::tdestOfModule(module)];

      __syncthreads();                  // Done with the previous module's map
      for (auto slot = threadIdx.x; slot < Jungfrau::PacketNum; slot += blockDim.x) {
        slotToPacket[slot] = NotPresent;
      }
      __syncthreads();

      auto const __restrict__ outMod = &pyld.out[module * Jungfrau::ModulePixels];

      if (modSub.size == 0) {           // Module absent: zero its whole frame
        for (auto i = tid; i < Jungfrau::ModulePixels; i += stride)  outMod[i] = 0.f;
        continue;
      }

      // The nested batch begins at the module sub-frame's first byte
      auto const __restrict__ nested = (Drp::Gpu::EvtBatcherHeader const*)modSub.data(pyld.data);
      auto const lineWidth = nested->lineWidth();
      auto const paddedPkt = (Drp::JungfrauData::PacketSize + lineWidth - 1) & ~(lineWidth - 1);
      auto const nestedSz  = lineWidth + Jungfrau::PacketNum * (paddedPkt + lineWidth);
      if (modSub.size < nestedSz) {     // Not the expected nested layout
        for (auto i = tid; i < Jungfrau::ModulePixels; i += stride)  outMod[i] = 0.f;
        continue;
      }
      auto const __restrict__ pktBase = (uint8_t const*)nested + lineWidth;
      auto const              pktPitch = paddedPkt + lineWidth;

      // Build the inverse map, one thread per packet.  The header's packetnum
      // says where the packet's pixels belong, as on the CPU side; a value
      // outside the frame means an extra packet, which is dropped.
      for (auto pkt = threadIdx.x; pkt < Jungfrau::PacketNum; pkt += blockDim.x) {
        auto const __restrict__ tail =
          (Drp::Gpu::EvtBatcherSubFrameTail const*)(pktBase + pkt * pktPitch + paddedPkt);
        if (tail->size() != Drp::JungfrauData::PacketSize)  continue;  // Wrong size: drop
        // packetnum is a uint32 in a 2-byte-packed header, so read it as two
        // uint16s to stay aligned whatever the line width
        auto const __restrict__ halves = (uint16_t const*)(pktBase + pkt * pktPitch + pktNumOff);
        auto const pktNum = unsigned(halves[0]) | (unsigned(halves[1]) << 16);
        if (pktNum < Jungfrau::PacketNum)  slotToPacket[pktNum] = uint8_t(pkt);
      }
      __syncthreads();

      // Scatter and calibrate.  Iterating over output slots means a missing
      // packet is zeroed as it is met, with no separate clearing pass.
      auto const rangeMask{(1u << Jungfrau::RangeBits) - 1u};
      auto const dataMask {(1u << Jungfrau::RangeOffset) - 1u};
      for (auto i = tid; i < Jungfrau::ModulePixels; i += stride) {
        auto const slot = i / pixPerPkt;
        auto const pix  = i - slot * pixPerPkt;
        auto const pkt  = slotToPacket[slot];
        if (pkt == NotPresent) {        // Packet never arrived
          outMod[i] = 0.f;
          continue;
        }
        auto const __restrict__ raw =
          (uint16_t const*)(pktBase + pkt * pktPitch + sizeof(Drp::JungfrauData::Header));
        auto const value = raw[pix];
        auto const range = (value >> Jungfrau::RangeOffset) & rangeMask;
        auto const data  = value & dataMask;
        auto const idx   = module * Jungfrau::ModulePixels + i;
        outMod[i] = (float(data) - peds[range * nPixels + idx]) * gains[range * nPixels + idx];
      }
    }
  }
};

// Instantiating the kernel template here puts the interpretation in the same CUDA
// module as the kernel, so it inlines.  See ReaderKernels.cuh.
void Jungfrau::recordEvent(cudaStream_t           stream,
                           unsigned               blocks,
                           unsigned               threads,
                           const EventKernelArgs& args)
{
  JungfrauCalib const calib{m_pedsVec_d, m_gainsVec_d, m_nModules, nPixels()};
  _event<JungfrauCalib><<<blocks, threads, 0, stream>>>(args, calib);
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new Jungfrau(para, pool);
}

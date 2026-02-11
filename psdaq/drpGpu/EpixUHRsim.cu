#include "EpixUHRsim.hh"

#include "GpuAsyncLib.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"
#include "SimDetector.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

struct uhr_domain{ static constexpr char const* name{"EpixUHRsim"}; };
using uhr_scoped_range = nvtx3::scoped_range_in<uhr_domain>;


namespace Drp {
  class PGPEvent;
  namespace Gpu {

// The functionality of the Drp::Detector is need to set up each of the panels
// However, the data description and handling in the GPU case will be different,
// so we create a Drp Detector class to handle just the protion we need.
// Derive from Drp::SimDetector so it can be made non-abstract.
class SimDet : public SimDetector
{
public:
  SimDet(Parameters* para, MemPoolGpu* pool, unsigned len=100) : SimDetector(para, pool, len) {}
  using SimDetector::event;
  void event(Dgram&, const void* bufEnd, PGPEvent*, uint64_t count) override { /* Not used */ }

  unsigned rangeOffset() const override { return 0; /* Not used */ }
  unsigned rangeBits()   const override { return 0; /* Not used */ }

  void recordGraph(cudaStream_t          stream,
                   const unsigned&       index,
                   const unsigned        panel,
                   uint16_t const* const data) override { /* Not used */ }
protected:
  size_t _genL1Payload(uint8_t* buffer, size_t bufSize) override
  {
    memset(buffer, 0x5a, bufSize);      // Some junk for now
    return bufSize;
  }
};

class RawDef : public VarDef
{
public:
  enum index
    {
      array_raw
    };

  RawDef()
  {
    Alg raw("raw", 0, 0, 0);
    NameVec.push_back({"array_raw", Name::UINT8, 1});
  }
};
  } // Gpu
} // Drp


EpixUHRsim::EpixUHRsim(Drp::Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  _initialize<Drp::Gpu::SimDet>(para, pool);

  // Check there is enough space in the DMA buffers for this many pixels
  assert(NPixels <= (pool.dmaSize() - sizeof(DmaDsc) - sizeof(TimingHeader)) / sizeof(uint16_t));

  // Set up buffers
  auto nPanels = pool.panels().size();
  pool.createCalibBuffers(nPanels, NPixels);

  // Create device pedestal and gain arrays of size nPanels
  chkError(cudaMalloc(&m_pedArr_d,  nPanels * sizeof(*m_pedArr_d)));
  chkError(cudaMalloc(&m_gainArr_d, nPanels * sizeof(*m_gainArr_d)));

  // Allocate space for the calibration constants for each panel
  m_pedsVec_d.resize(nPanels);
  m_gainsVec_d.resize(nPanels);
  for (unsigned i = 0; i < nPanels; ++i) {
    chkError(cudaMalloc(&m_pedsVec_d[i],  NRanges * NPixels * sizeof(*m_pedsVec_d[i])));
    chkError(cudaMalloc(&m_gainsVec_d[i], NRanges * NPixels * sizeof(*m_gainsVec_d[i])));

    // Fill the device arrays with the device pointers
    chkError(cudaMemcpy(&m_pedArr_d[i],  &m_pedsVec_d[i],  sizeof(m_pedsVec_d[i]),  cudaMemcpyHostToDevice));
    chkError(cudaMemcpy(&m_gainArr_d[i], &m_gainsVec_d[i], sizeof(m_gainsVec_d[i]), cudaMemcpyHostToDevice));
  }
}

EpixUHRsim::~EpixUHRsim()
{
  printf("*** EpixUHRsim dtor 1\n");
  chkError(cudaFree(m_gainArr_d));
  chkError(cudaFree(m_pedArr_d));
  auto pool = m_pool->getAs<MemPoolGpu>();
  for (unsigned i = 0; i < pool->panels().size(); ++i) {
    chkError(cudaFree(m_gainsVec_d[i]));
    chkError(cudaFree(m_pedsVec_d[i]));
  }
  printf("*** EpixUHRsim dtor 2\n");

  pool->destroyCalibBuffers();
  printf("*** EpixUHRsim dtor 3\n");
}

unsigned EpixUHRsim::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::EpixUHRsim configure");

  // Only 1 panel in the simulator
  unsigned panel{0};
  unsigned rc = m_dets[panel]->configure(config_alias, xtc, bufEnd);
  printf("*** Gpu::EpixUHRsim configure for %u done: rc %d, sz %u\n", panel, rc, xtc.sizeofPayload());
  if (rc) {
    logging::error("Gpu::EpixUHRsim::configure failed for %s\n", m_params[panel].device);
  }

  Alg alg("raw", 0, 0, 0);
  NamesId namesId(nodeId, EventNamesIndex); // + panel);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para->detName.c_str(), alg,
                                         m_para->detType.c_str(), m_para->serNo.c_str(), namesId, m_para->detSegment);
  RawDef dataDef;
  names.add(xtc, bufEnd, dataDef);
  m_namesLookup[namesId] = NameIndex(names);

  logging::info("Gpu::EpixUHRsim configure: xtc size %u", xtc.sizeofPayload());

  return rc;
}

unsigned EpixUHRsim::beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
  // Only 1 panel in the simulator
  unsigned rc = m_dets[0]->beginrun(xtc, bufEnd, info);
  if (rc) {
    logging::error("Gpu::EpixUHRsim::beginrun failed for %s\n", m_params[0].device);
  }

  // Load the calibration constants onto the GPU
  // @todo: Fetch calibration constants for each panel
  std::vector<float> peds(NPixels, 0.0);
  std::vector<float> gains(NPixels, 1.0);
  auto pool = m_pool->getAs<MemPoolGpu>();
  for (unsigned i = 0; i < pool->panels().size(); ++i) {
    auto peds_d  = m_pedsVec_d[i];
    auto gains_d = m_gainsVec_d[i];
    for (unsigned range = 0; range < NRanges; ++range) {
      chkError(cudaMemcpy(peds_d,  peds.data(),  NPixels * sizeof(*peds_d),  cudaMemcpyHostToDevice));
      chkError(cudaMemcpy(gains_d, gains.data(), NPixels * sizeof(*gains_d), cudaMemcpyHostToDevice));
      peds_d  += NPixels;
      gains_d += NPixels;
    }
  }

  return rc;
}

void EpixUHRsim::issuePhase2(TransitionId::Value tid)
{
  m_dets[0]->gpuDetector()->issuePhase2(tid);
}


void EpixUHRsim::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent*, uint64_t count)
{
  logging::info("Gpu::EpixUHRsim event");

  // @todo: Deal with prescaled raw or calibrated data for each panel here?
}

//__device__ void EpixUHRsim::calibrate(float*    const __restrict__ calib,
//                                      uint16_t* const __restrict__ raw,
//                                      unsigned  const              count,
//                                      unsigned  const              nPanels) const
//{
//  auto const tid     = blockIdx.x * blockDim.x + threadIdx.x;
//  auto const stride  = gridDim.x * blockDim.x;
//  if (tid == 0)  printf("*** tid %d, stride %d, nPanels %u\n", tid, stride, nPanels);
//  auto const panel   = tid / (stride / nPanels);  // Split the panel handling evenly across the allocated threads
//  if (tid == 0)  printf("*** panel %u\n", panel);
//  constexpr auto nPixels = count / nPanels;
//
//  if (tid == 0)  printf("*** m_pedArr_d %p\n", m_pedArr_d);
//  auto const pedArr  = m_pedArr_d[panel];
//  if (tid == 0)  printf("*** pedArr %p\n", pedArr);
//  if (tid == 0)  printf("*** m_gainArr_d %p\n", m_gainArr_d);
//  auto const gainArr = m_gainArr_d[panel];
//  if (tid == 0)  printf("*** gainArr %p\n", gainArr);
//  if (tid == 0)  printf("*** count %u\n", count);
//  for (auto i = tid; i < count; i += stride) {
//    auto const range = (raw[i] >> RangeOffset) & ((1 << RangeBits) - 1);
//    auto const peds  = &pedArr [range * nPixels];
//    auto const gains = &gainArr[range * nPixels];
//    auto const data  = raw[i] & ((1 << RangeOffset) - 1);
//    calib[panel * nPixels + i] = (data - peds[i]) * gains[i];
//  }
//  printf("*** calibrate returning\n");
//}

// This kernel performs the data calibration
static __global__ void _calibrate(float*   const        __restrict__ calibBuffers,
                                  size_t   const                     calibBufsCnt,
                                  uint16_t const* const __restrict__ in,
                                  unsigned const&                    index,
                                  unsigned const                     panel,
                                  float    const* const __restrict__ peds_,
                                  float    const* const __restrict__ gains_)
{
#if 0
  // Place the calibrated data for a given panel in the calibBuffers array at the appropriate offset
  auto const __restrict__ out = &calibBuffers[index * calibBufsCnt + panel * EpixUHRsim::NPixels];
  int stride = gridDim.x * blockDim.x;
  int pixel  = blockIdx.x * blockDim.x + threadIdx.x;

  // @todo: Pass these arrays of nGains pointers in
  const float* const* __restrict__ peds [1 << EpixUHRsim::RangeBits];
  const float* const* __restrict__ gains[1 << EpixUHRsim::RangeBits];
  //  #pragma unroll ?
  for (unsigned i = 0; i < 1 << EpixUHRsim::RangeBits; ++i) {
    peds[i]  = &peds_ [i * EpixUHRsim::NPixels];
    gains[i] = &gains_[i * EpixUHRsim::NPixels];
  }

  __shared__ uint8_t gnMask[stride/warpSize];    // or 2048/32: discover this
  __shared__ float   sPeds[EpixUHRsim::RangeBits][stride];
  __shared__ float   sGains[EpixUHRsim::RangeBits][stride];

  constexpr auto gm{(1 << EpixUHRsim::RangeBits) - 1};
  gnMask[0] = gm;
  for (int i = pixel; i < EpixUHRsim::NPixels; gnMask[i/warpSize] = gm, i += stride) {
    const auto data  = in[i];
    const auto gain  = (data >> EpixUHRsim::RangeOffset) & gm;
    data &= (1 << EpixUHRsim::RangeOffset) - 1;
    const auto wid = i / warpSize;
    if (gnMask[wid] & (1 << gain)) {
      sPeds[i]  = peds[gain][i];
      sGains[i] = gains[gain][i];
      gnMask[wid] ^= 1 << gain;         // Need atomic here
    }
    out[i] = (float(data) - sPeds[gain][i]) * sGains[gain][i];
  }
#endif
}

// This routine records the graph that calibrates the data
void EpixUHRsim::recordGraph(cudaStream_t          stream,
                             const unsigned&       index_d,
                             const unsigned        panel,
                             uint16_t const* const rawBuffer)
{
  uhr_scoped_range r{/*"EpixUHRsim::recordGraph"*/}; // Expose function name via NVTX

  auto pool    = m_pool->getAs<MemPoolGpu>();
  auto nPanels = pool->panels().size();

  // Check that panel is within range
  assert (panel < nPanels);

  // @todo: want 3 blocks of 512 threads to handle 126 pixels each
  unsigned   chunks{128};               // Number of pixels handled per thread
  unsigned   tpb   {256};               // Threads per block
  unsigned   bpg   {(NPixels + chunks * tpb - 1) / (chunks * tpb)}; // Blocks per grid
  const auto peds         = m_pedsVec_d[panel];
  const auto gains        = m_gainsVec_d[panel];
  auto const calibBuffers = pool->calibBuffers_d();
  auto const calibBufsCnt = pool->calibBufsSize() / sizeof(*calibBuffers);
  _calibrate<<<bpg, tpb, 0, stream>>>(calibBuffers, calibBufsCnt, rawBuffer, index_d, panel, peds, gains);
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHRsim(para, pool);
}

#include "EpixUHRemu.hh"

#include "GpuAsyncLib.hh"
#include "psdaq/service/EbDgram.hh"
#include "psalg/utils/SysLog.hh"
#include "drp/drp.hh"                   // For PGPEvent

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

EpixUHRemu::EpixUHRemu(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Call common code to set up a vector of Drp::EpixUHRemus
  _initialize<Drp::AreaDetector>(para, pool);

  // Check there is enough space in the DMA buffers for this many pixels
  assert(NPixels <= (pool.dmaSize() - sizeof(DmaDsc) - sizeof(TimingHeader)) / sizeof(uint16_t));

  // Set up buffers
  pool.createCalibBuffers(m_pool->nbuffers(), m_dets.size(), NPixels);

  // Allocate space for the calibration constants for each panel
  m_peds_h.resize(m_dets.size());
  m_gains_h.resize(m_dets.size());
  for (unsigned i = 0; i < m_dets.size(); ++i) {
    chkError(cudaMalloc(&m_peds_h[i],  NGains * NPixels * sizeof(*m_peds_h[i])));
    chkError(cudaMalloc(&m_gains_h[i], NGains * NPixels * sizeof(*m_gains_h[i])));
  }
}

EpixUHRemu::~EpixUHRemu()
{
  printf("*** EpixUHRemu dtor 1\n");
  for (unsigned i = 0; i < m_dets.size(); ++i) {
    chkError(cudaFree(m_peds_h[i]));
    chkError(cudaFree(m_gains_h[i]));
  }
  printf("*** EpixUHRemu dtor 2\n");

  printf("*** EpixUHRemu dtor 3\n");

  auto pool = m_pool->getAs<MemPoolGpu>();
  pool->destroyCalibBuffers();
  printf("*** EpixUHRemu dtor 4\n");
}


unsigned EpixUHRemu::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  // Do beginRun for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->beginrun(xtc, bufEnd, runInfo);
    if (rc) {
      logging::error("Gpu::EpixUHRemu::beginrun failed for %s\n", m_params[i].device);
      break;
    }
  }

  // @todo: Fetch calibration constants for each panel and store them on the GPU
  std::vector<float> peds(NPixels, 0.0);
  std::vector<float> gains(NPixels, 1.0);
  for (unsigned i = 0; i < m_dets.size(); ++i) {
    auto peds_d  = m_peds_h[i];
    auto gains_d = m_gains_h[i];
    for (unsigned gn = 0; gn < NGains; ++gn) {
      chkError(cudaMemcpy(peds_d,  peds.data(),  NPixels * sizeof(*peds_d),  cudaMemcpyHostToDevice));
      chkError(cudaMemcpy(gains_d, gains.data(), NPixels * sizeof(*gains_d), cudaMemcpyHostToDevice));
      peds_d  += NPixels * sizeof(*peds_d);
      gains_d += NPixels * sizeof(*gains_d);
    }
  }

  return rc;
}

// This kernel performs the data calibration
static __global__ void _calibrate(float*   const        __restrict__ calibBuffers,
                                  uint16_t const* const __restrict__ in,
                                  const unsigned&                    index,
                                  const unsigned                     panel,
                                  const unsigned                     nPanels,
                                  float* const         __restrict__  peds_,
                                  float* const         __restrict__  gains_)
{
  int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  auto const __restrict__ out = &calibBuffers[index * nPanels * EpixUHRemu::NPixels];

  const auto gainMask = (1 << EpixUHRemu::GainOffset) - 1;
  for (int i = pixel; i < EpixUHRemu::NPixels; i += blockDim.x * gridDim.x) {
    const auto gain     = (in[i] >> EpixUHRemu::GainOffset) & ((1 << EpixUHRemu::GainBits) - 1);
    const auto peds     = peds_  + gain * EpixUHRemu::NPixels;
    const auto gains    = gains_ + gain * EpixUHRemu::NPixels;
    const auto data     = in[i] & gainMask;
    out[i] = (float(data) - peds[i]) * gains[i];
  }
}

// This routine records the graph that calibrates the data
void EpixUHRemu::recordGraph(cudaStream_t&                      stream,
                             const unsigned&                    index_d,
                             const unsigned                     panel,
                             uint16_t const* const __restrict__ rawBuffer)
{
  auto nPanels = m_dets.size();

  // Check that panel is within range
  assert (panel < nPanels);

  int threads = 1024;
  int blocks  = (NPixels + threads-1) / threads; // @todo: Limit this?
  auto       pool         = m_pool->getAs<MemPoolGpu>();
  const auto peds         = m_peds_h[panel];
  const auto gains        = m_gains_h[panel];
  auto const calibBuffers = pool->calibBuffers() + panel * NPixels;
  _calibrate<<<blocks, threads, 0, stream>>>(calibBuffers, rawBuffer, index_d, panel, nPanels, peds, gains);
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHRemu(para, pool);
}

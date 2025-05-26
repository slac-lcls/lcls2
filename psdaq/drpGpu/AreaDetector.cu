#include "AreaDetector.hh"

#include "GpuAsyncLib.hh"
#include "psdaq/service/EbDgram.hh"
#include "psalg/utils/SysLog.hh"
#include "drp/drp.hh"                   // For PGPEvent

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

AreaDetector::AreaDetector(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Call common code to set up a vector of Drp::AreaDetectors
  _initialize<Drp::AreaDetector>(para, pool);

  // Use a non-generic hack to determine the number of pixels
  // sim_length is in units of uint32_ts, so 2 pixels per count
  m_nPixels = para.kwargs.find("sim_length") != para.kwargs.end()
            ? std::stoul(para.kwargs["sim_length"]) * 2
            : 1024;                     // @todo: revisit

  // Check there is enough space in the DMA buffers for this many pixels
  assert(m_nPixels <= (pool.dmaSize() - sizeof(DmaDsc) - sizeof(TimingHeader)) / sizeof(uint16_t));

  // Set up buffers
  pool.createCalibBuffers(m_pool->nbuffers(), m_dets.size(), m_nPixels);
}

AreaDetector::~AreaDetector()
{
  printf("*** AreaDetector dtor 1\n");
  auto pool = m_pool->getAs<MemPoolGpu>();
  pool->destroyCalibBuffers();
  printf("*** AreaDetector dtor 4\n");
}

// This kernel performs the data calibration
static __global__ void _calibrate(float*   const        __restrict__ calibBuffers,
                                  uint16_t const* const __restrict__ in,
                                  const unsigned&                    index,
                                  const unsigned                     nPanels,
                                  const unsigned                     nPixels)
{
  int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  auto const __restrict__ out = &calibBuffers[index * nPanels * nPixels];

  for (int i = pixel; i < nPixels; i += blockDim.x * gridDim.x) {
    out[i] = float(in[i]);
  }
}

// This routine records the graph that calibrates the data
void AreaDetector::recordGraph(cudaStream_t&                      stream,
                               const unsigned&                    index_d,
                               const unsigned                     panel,
                               uint16_t const* const __restrict__ rawBuffer)
{
  printf("*** AreaDetector record: 1\n");
  auto nPanels = m_dets.size();

  // Check that panel is within range
  assert (panel < nPanels);
  printf("*** AreaDetector record: 2\n");

  int threads = 1024;
  int blocks  = (m_nPixels + threads-1) / threads; // @todo: Limit this?
  auto       pool         = m_pool->getAs<MemPoolGpu>();
  auto const calibBuffers = pool->calibBuffers() + panel * m_nPixels;
  _calibrate<<<blocks, threads, 0, stream>>>(calibBuffers, rawBuffer, index_d, nPanels, m_nPixels);
  printf("*** AreaDetector record: 5\n");
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new AreaDetector(para, pool);
}

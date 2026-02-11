#include "AreaDetector.hh"

#include "GpuAsyncLib.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

struct ad_domain{ static constexpr char const* name{"AreaDetector"}; };
using ad_scoped_range = nvtx3::scoped_range_in<ad_domain>;


namespace Drp {
  class PGPEvent;
  namespace Gpu {

// The functionality of the Drp::Detector is needed to set up each of the panels
// However, the data description and handling in the GPU case will be different,
// so we create a Drp Detector class to handle just the portion we need.
// Derive from Drp::XpmDetector so it can be made non-abstract.
class XpmDetector : public Drp::XpmDetector
{
public:
  XpmDetector(Parameters* para, MemPool* pool, unsigned len=100) : Drp::XpmDetector(para, pool, len) {}
  using Drp::XpmDetector::event;
  void event(Dgram&, const void* bufEnd, PGPEvent*, uint64_t count) override { /* Not used */ }
};

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


AreaDetector::AreaDetector(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Call common code to set up a vector of Drp::XpmDetectors
  _initialize<Drp::Gpu::XpmDetector>(para, pool);

  // Use a non-generic hack to determine the number of pixels
  // sim_length is in units of uint32_ts, so 2 pixels per count
  m_nPixels = para.kwargs.find("sim_length") != para.kwargs.end()
            ? std::stoul(para.kwargs["sim_length"]) * sizeof(uint32_t) / 2
            : 1024;                     // @todo: revisit

  // Check there is enough space in the DMA buffers for this many pixels
  assert(m_nPixels <= (pool.dmaSize() - sizeof(DmaDsc) - sizeof(TimingHeader)) / sizeof(uint16_t));

  // Set up buffers
  pool.createCalibBuffers(m_dets.size(), m_nPixels);
}

AreaDetector::~AreaDetector()
{
  printf("*** AreaDetector dtor 1\n");
  auto pool = m_pool->getAs<MemPoolGpu>();
  pool->destroyCalibBuffers();
  printf("*** AreaDetector dtor 2\n");
}

unsigned AreaDetector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::AreaDetector configure");

  // Configure the XpmDetector for each panel in turn
  // @todo: Do we really want to extend the Xtc for each panel, or does one speak for all?
  unsigned panel = 0;
  for (const auto& det : m_dets) {
    if (det->configure(config_alias, xtc, bufEnd)) {
      logging::error("Gpu::AreaDetector::configure failed for %s\n", m_params[panel].device);
      break;
    }
    ++panel;
  }

#if 0  // @todo: Deal with prescaled raw or calibrated data for each panel here?
  Alg alg("raw", 0, 0, 0);
  NamesId namesId(nodeId, EventNamesIndex + panel);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para->detName.c_str(), alg,
                                         m_para->detType.c_str(), m_para->serNo.c_str(), namesId, m_para->detSegment);
  RawDef dataDef;
  names.add(xtc, bufEnd, dataDef);
  m_namesLookup[namesId] = NameIndex(names);

  logging::info("Gpu::AreaDetector configure: xtc size %u", xtc.sizeofPayload());
#endif

  return 0;
}

void AreaDetector::event(Dgram& dgram, const void* bufEnd, PGPEvent*, uint64_t count)
{
  logging::info("Gpu::AreaDetector event");

  // @todo: Deal with prescaled raw or calibrated data for each panel here?
}

//__device__ void AreaDetector::calibrate(float*    const calib,
//                                        uint16_t* const raw,
//                                        unsigned  const count,
//                                        unsigned  const nFpgas) const
//{
//  auto const tid    = blockDim.x * blockIdx.x + threadIdx.x;
//  auto const stride = gridDim.x * blockDim.x;
//  auto const fpga   = tid / (stride / nFpgas);  // Split the FPGA handling evenly across the allocated threads
//
//  for (auto i = tid; i < count; i += stride) {
//    calib[fpga * nFpgas + i] = float(raw[i]);
//  }
//}

// This kernel performs the data calibration
static __global__ void _calibrate(float*   const        __restrict__ calibBuffers,
                                  const size_t                       calibBufsCnt,
                                  uint16_t const* const __restrict__ in,
                                  const unsigned&                    index,
                                  const unsigned                     panel,
                                  const unsigned                     nPixels)
{
  auto const __restrict__ out = &calibBuffers[index * calibBufsCnt + panel * nPixels];
  int stride = gridDim.x * blockDim.x;
  int pixel  = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = pixel; i < nPixels; i += stride) {
    out[i] = float(in[i]);
  }
}

// This routine records the graph that calibrates the data
void AreaDetector::recordGraph(cudaStream_t          stream,
                               const unsigned&       index_d,
                               const unsigned        panel,
                               uint16_t const* const rawBuffer)
{
  ad_scoped_range r{/*"AreaDetector::recordGraph"*/}; // Expose function name via NVTX

  auto nPanels = m_dets.size();

  // Check that panel is within range
  assert (panel < nPanels);

  unsigned   chunks{128};               // Number of pixels handled per thread
  unsigned   tpb   {256};               // Threads per block
  unsigned   bpg   {(m_nPixels + chunks * tpb - 1) / (chunks * tpb)}; // Blocks per grid
  auto       pool         = m_pool->getAs<MemPoolGpu>();
  auto const calibBuffers = pool->calibBuffers_d();
  auto const calibBufsCnt = pool->calibBufsSize() / sizeof(*calibBuffers);
  _calibrate<<<bpg, tpb, 0, stream>>>(calibBuffers, calibBufsCnt, rawBuffer, index_d, panel, m_nPixels);
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new AreaDetector(para, pool);
}

#include "AreaDetector.hh"

#include "ReaderKernels.cuh"

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

// The functionality of the Drp::Detector is needed to set up each panel.
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
  pool.createCalibBuffers(m_nPixels);
}

AreaDetector::~AreaDetector()
{
  auto pool = m_pool->getAs<MemPoolGpu>();
  pool->destroyCalibBuffers();
}

unsigned AreaDetector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::AreaDetector configure");

  // Configure the XpmDetector for the panel
  // @todo: Do we really want to extend the Xtc for each panel, or does one speak for all?
  if (m_det->configure(config_alias, xtc, bufEnd)) {
    logging::error("Gpu::AreaDetector::configure failed for %s\n", m_para->device);
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

// Instantiating the kernel template here puts the calibration in the same CUDA
// module as the kernel, so it inlines.  See ReaderKernels.cuh.
void AreaDetector::recordEvent(cudaStream_t           stream,
                               unsigned               blocks,
                               unsigned               threads,
                               const EventKernelArgs& args)
{
  // No reference buffers: only the simulator controls the raw data it generates,
  // so only it can supply something to verify the calibration against
  PedGainCalib const calib{pedestals_d(),
                           gains_d(),
                           nullptr,
                           0,
                           rangeOffset(),
                           rangeBits()};
  _event<PedGainCalib><<<blocks, threads, 0, stream>>>(args, calib);
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new AreaDetector(para, pool);
}

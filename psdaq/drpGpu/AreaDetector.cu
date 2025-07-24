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


namespace Drp {
  class PGPEvent;
  namespace Gpu {

// The functionality of the Drp::Detector is needed to set up each of the panels
// However, the data description and handling in the GPU case will be different,
// so we create a Drp Detector class to handle just the protion we need.
// Derive from Drp::XpmDetector so it can be made non-abstract.
class XpmDetector : public Drp::XpmDetector
{
public:
  XpmDetector(Parameters* para, MemPool* pool, unsigned len=100) : Drp::XpmDetector(para, pool, len) {}
  using Drp::XpmDetector::event;
  void event(Dgram& dgram, const void* bufEnd, Drp::PGPEvent* event, uint64_t count) override { /* Not used */ }
};

class FexDef : public VarDef
{
public:
  enum index
  {
    fex
  };

  FexDef()
  {
    Alg fex("fex", 0, 0, 0);
    NameVec.push_back({"fex", Name::UINT8, 1});
  }
};
  } // Gpu
} // Drp


AreaDetector::AreaDetector(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Call common code to set up a vector of Drp::AreaDetectors
  //_initialize<Drp::AreaDetector>(para, pool);
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
  printf("*** AreaDetector dtor 4\n");
}

unsigned AreaDetector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::AreaDetector configure");
  unsigned rc = 0;

  // Configure the XpmDetector for each panel in turn
  // @todo: Do we really want to extend the Xtc for each panel, or does one speak for all?
  unsigned i = 0;
  for (const auto& det : m_dets) {
    printf("*** Gpu::AreaDetector configure for %u start\n", i);
    rc = det->configure(config_alias, xtc, bufEnd);
    printf("*** Gpu::AreaDetector configure for %u done: rc %d, sz %u\n", i, rc, xtc.sizeofPayload());
    if (rc) {
      logging::error("Gpu::AreaDetector::configure failed for %s\n", m_params[i].device);
      break;
    }
    ++i;
  }

  Alg fexAlg("fex", 0, 0, 0);
  NamesId fexNamesId(nodeId, FexNamesIndex);
  Names& fexNames = *new(xtc, bufEnd) Names(bufEnd,
                                            m_para->detName.c_str(), fexAlg,
                                            m_para->detType.c_str(), m_para->serNo.c_str(), fexNamesId, m_para->detSegment);
  FexDef myFexDef;
  fexNames.add(xtc, bufEnd, myFexDef);
  m_namesLookup[fexNamesId] = NameIndex(fexNames);

  logging::info("Gpu::AreaDetector configure: xtc size %u", xtc.sizeofPayload());

  return 0;
}

size_t AreaDetector::event(Dgram& dgram, const void* bufEnd, unsigned payloadSize)
{
  // The Dgram header is constructed in the pebble buffer, and this buffer is
  // not used to hold all of the data.  However, bufEnd needs to point to a
  // location that makes it seem like the buffer is big enough to contain both
  // the header and data so that the Xtc allocate in fex.set_array_shape() can
  // succeed.  This may be larger than the pebble buffer and we therefore must
  // be careful not to write beyond its end.  This is checked for below.
  logging::info("Gpu::AreaDetector event: dg %p, extent %u, size %u", &dgram, dgram.xtc.extent, payloadSize);

  // FEX is Reduced data
  NamesId fexNamesId(nodeId, FexNamesIndex);
  CreateData fex(dgram.xtc, bufEnd, m_namesLookup, fexNamesId);

  // CreateData places into the Dgram, in one contiguous block:
  // - the ShapesData Xtc
  // - the Shapes Xtc with its payload
  // - the Data Xtc (the payload of which is on the GPU)
  // Measure the size of the header block
  auto headerSize = (uint8_t*)dgram.xtc.next() - (uint8_t*)&dgram;
  printf("*** Gpu::AreaDetector event: payloadSz %u, length %p - %p = %zd\n",
         dgram.xtc.sizeofPayload(), dgram.xtc.next(), &dgram, headerSize);

  auto pool = m_pool->getAs<MemPoolGpu>();
  // Make sure the header will fit in the space reserved for it on the GPU
  if (size_t(headerSize) > pool->reduceBufReserved()) {
    logging::critical("Header is too large (%zu) for reduce buffer's reserved space (%zu)",
                      headerSize, pool->reduceBufReserved());
    abort();
  }
  // Make sure the header fits in the pebble buffer
  if (size_t(headerSize) > pool->pebble.bufferSize()) {
    logging::critical("Header is too large (%zu) for pebble buffer (%zu)",
                      headerSize, pool->pebble.bufferSize());
    abort();
  }

  // Update the header with the size and shape of the data payload
  // This does not write beyond headerSize bytes into the pebble buffer
  unsigned fex_shape[MaxRank] = { payloadSize };
  fex.set_array_shape(FexDef::fex, fex_shape);
  return headerSize;
}

// This kernel performs the data calibration
static __global__ void _calibrate(float*   const* const __restrict__ calibBuffers,
                                  uint16_t const* const __restrict__ in,
                                  const unsigned&                    index,
                                  const unsigned                     panel,
                                  const unsigned                     nPixels)
{
  int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  auto const __restrict__ out = &calibBuffers[index][panel * nPixels];

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
  printf("*** AreaDetector record: 3, panel %u, threads %d, blocks %d, nPixels %d\n", panel, threads, blocks, m_nPixels);
  auto       pool         = m_pool->getAs<MemPoolGpu>();
  auto const calibBuffers = pool->calibBuffers_d();
  printf("*** AreaDetector record: 4, calibBufSz %zu\n", pool->calibBufSize());
  _calibrate<<<blocks, threads, 0, stream>>>(calibBuffers, rawBuffer, index_d, panel, m_nPixels);
  printf("*** AreaDetector record: 5\n");
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new AreaDetector(para, pool);
}

#include "EpixUHRemu.hh"

#include "ReaderKernels.cuh"

#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"
#include "drp/XpmDetector.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

struct uhr_domain{ static constexpr char const* name{"EpixUHRemu"}; };
using uhr_scoped_range = nvtx3::scoped_range_in<uhr_domain>;


namespace Drp {
  class PGPEvent;
  namespace Gpu {

// The functionality of the Drp::Detector is needed to set up the panel.
// However, the data description and handling in the GPU case will be different,
// so we create a Drp Detector class to handle just the portion we need.
// Derive from Drp::XpmDetector so it can be made non-abstract.
class XpmDetector : public Drp::XpmDetector
{
public:
  XpmDetector(Parameters* para, MemPool* pool, unsigned len=100) :
    Drp::XpmDetector(para, pool, len),
    m_pyModule(PyImport_ImportModule("psdaq.configdb.epixuhremu_config"))
  {
    // Imported here rather than in connectionInfo(), which is called once per
    // Allocate, and following Drp::XpmDetector's own pattern: import once, resolve
    // the function from the module dict per call.
    if (!m_pyModule) {
      PyErr_Print();
      logging::error("Gpu::EpixUHRemu: cannot import epixuhremu_config; LCLS-II "
                     "timing will have to be configured by hand");
    }
  }
  using Drp::XpmDetector::event;
  void event(Dgram&, const void* bufEnd, PGPEvent*, uint64_t count) override { /* Not used */ }

  // The emulator firmware needs LCLS-II timing configured, which the real
  // detectors do not and which is otherwise done by hand from the devGui.
  //
  // This is the least invasive place for it.  Drp::XpmDetector's Python hook is
  // hardwired to psdaq.configdb.xpmdet_config (XpmDetector.cc:37), and changing
  // that, or xpmdet_config itself, would put the CPU DRPs at risk for the sake of
  // a detector that will never run in production.  But nothing stops a second,
  // independent import from GPU-only code: epixuhremu_config reaches the rogue
  // tree through xpmdet_config's own module global, so neither has to change.
  //
  // The GIL is already held here -- PGPDetectorApp::connectionInfo wraps
  // m_det->connectionInfo() in PY_ACQUIRE_GIL_GUARD -- so no guard is needed.
  json connectionInfo(const json& msg) override
  {
    // Before the base call, not after.  With the timing link down,
    // xpmdet_connectionInfo() reads the XPM remote link id as 0xffffffff and raises
    // 'Illegal XPM Remote link id', so a hook after it never runs -- and the link
    // being down is precisely the case this exists to fix.  Its own RxPllReset retry
    // does not recover it; ConfigLclsTimingV2() also clears UseMiniTpg and issues
    // TxPhyReset and the Tx and Rx user resets.
    if (m_pyModule) {
      auto dict = PyModule_GetDict(m_pyModule);            // Borrowed
      auto func = PyDict_GetItemString(dict, "epixuhremu_configTiming"); // Borrowed
      if (func) {
        auto rv = PyObject_CallObject(func, nullptr);       // Not CallFunction(f, ""),
        if (rv)  Py_DECREF(rv);                            // which passes None, not ()
        else     PyErr_Print();
      } else {
        logging::error("Gpu::EpixUHRemu: epixuhremu_config has no "
                       "epixuhremu_configTiming()");
      }
    }
    return Drp::XpmDetector::connectionInfo(msg);          // The whole xpmdet path
  }
private:
  PyObject* m_pyModule;
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

EpixUHRemu::EpixUHRemu(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Call common code to set up a vector of Drp::XpmDetectors
  _initialize<Drp::Gpu::XpmDetector>(para, pool);

  // Check there is enough space in the DMA buffers for this many pixels
  constexpr size_t minDmaSize{NPixels * sizeof(uint16_t) + sizeof(TimingHeader)};
  if (minDmaSize > pool.dmaSize()) {
    logging::critical("DMA buffer of %zu bytes is too small for %u pixels: need %zu",
                      pool.dmaSize(), NPixels, minDmaSize);
    abort();
  }

  // Set up buffers
  pool.createCalibBuffers(NPixels);

  // Allocate space for the calibration constants
  chkError(cudaMalloc(&m_pedsVec_d,  NRanges * NPixels * sizeof(*m_pedsVec_d)));
  chkError(cudaMalloc(&m_gainsVec_d, NRanges * NPixels * sizeof(*m_gainsVec_d)));
}

EpixUHRemu::~EpixUHRemu()
{
  auto pool = m_pool->getAs<MemPoolGpu>();
  if (m_gainsVec_d)  chkError(cudaFree(m_gainsVec_d));
  if (m_pedsVec_d)   chkError(cudaFree(m_pedsVec_d));

  pool->destroyCalibBuffers();
}

unsigned EpixUHRemu::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::EpixUHRemu configure");

  // Configure the XpmDetector for the panel
  unsigned rc = m_det->configure(config_alias, xtc, bufEnd);
  if (rc) {
    logging::error("Gpu::EpixUHRemu::configure failed for %s\n", m_para->device);
  }

  Alg alg("raw", 0, 0, 0);
  NamesId namesId(nodeId, EventNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para->detName.c_str(), alg,
                                         m_para->detType.c_str(), m_para->serNo.c_str(), namesId, m_para->detSegment);
  RawDef dataDef;
  names.add(xtc, bufEnd, dataDef);
  m_namesLookup[namesId] = NameIndex(names);

  logging::info("Gpu::EpixUHRemu configure: xtc size %u", xtc.sizeofPayload());

  return 0;
}

unsigned EpixUHRemu::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  unsigned rc = m_det->beginrun(xtc, bufEnd, runInfo);
  if (rc) {
    logging::error("Gpu::EpixUHRemu::beginrun failed for %s\n", m_para->device);
  }

  // Load the calibration constants onto the GPU
  // @todo: Fetch calibration constants
  std::vector<float> peds(NPixels, 0.0);
  std::vector<float> gains(NPixels, 1.0);
  auto pool = m_pool->getAs<MemPoolGpu>();
  auto peds_d  = m_pedsVec_d;
  auto gains_d = m_gainsVec_d;
  for (unsigned range = 0; range < NRanges; ++range) {
    chkError(cudaMemcpy(peds_d,  peds.data(),  NPixels * sizeof(*peds_d),  cudaMemcpyDefault));
    chkError(cudaMemcpy(gains_d, gains.data(), NPixels * sizeof(*gains_d), cudaMemcpyDefault));
    peds_d  += NPixels;
    gains_d += NPixels;
  }

  return rc;
}

void EpixUHRemu::event(Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count)
{
  constexpr uint32_t lane{0}; // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  size_t size = buffer->size;
  constexpr auto eventSize{sizeof(TimingHeader) + NPixels * sizeof(uint16_t)};
  if      (size  < eventSize)          dgram.xtc.damage.increase(Damage::MissingData);
  else if (size == m_pool->dmaSize())  dgram.xtc.damage.increase(Damage::Truncated);

  // @todo: Deal with prescaled raw for the panel here?
}

// Instantiating the kernel template here puts the calibration in the same CUDA
// module as the kernel, so it inlines.  See ReaderKernels.cuh.
void EpixUHRemu::recordEvent(cudaStream_t           stream,
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
  return new EpixUHRemu(para, pool);
}

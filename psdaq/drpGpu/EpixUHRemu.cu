#include "EpixUHRemu.hh"
#include "EpixUHRcalib.hh"

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
    // Import once, as Drp::XpmDetector does; connectionInfo() runs per Allocate
    if (!m_pyModule) {
      PyErr_Print();
      logging::error("Gpu::EpixUHRemu: cannot import epixuhremu_config; LCLS-II "
                     "timing will have to be configured by hand");
    }
  }
  using Drp::XpmDetector::event;
  void event(Dgram&, const void* bufEnd, PGPEvent*, uint64_t count) override { /* Not used */ }

  // The emulator firmware needs LCLS-II timing configured, which real detectors
  // don't.  A separate import here leaves xpmdet_config, shared with the CPU DRPs,
  // untouched.  The caller already holds the GIL.
  json connectionInfo(const json& msg) override
  {
    // Before the base call: with the link down, xpmdet_connectionInfo() raises
    // 'Illegal XPM Remote link id', and its RxPllReset retry doesn't recover it.
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

// The raw block: the panel's u16 pixels, rank 2 as Drp::EpixUHR3x2 declares them, so
// emulator and real files look the same offline
class RawU16Def : public VarDef
{
public:
  enum index
    {
      raw
    };

  RawU16Def()
  {
    NameVec.push_back({"raw", Name::UINT16, 2});
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
  logging::info("Gpu::EpixUHRemu configure: alias '%s'%s", config_alias.c_str(),
                m_passthru ? ", recording raw u16 uncalibrated and unreduced" : "");

  // Configure the XpmDetector for the panel
  unsigned rc = m_det->configure(config_alias, xtc, bufEnd);
  if (rc) {
    logging::error("Gpu::EpixUHRemu::configure failed for %s\n", m_para->device);
  }

  Alg alg("raw", 1, 0, 0);               // 1.0.0: [336][576], as ePixUHR3x2 raw 1.0.0
  NamesId namesId(nodeId, EventNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para->detName.c_str(), alg,
                                         m_para->detType.c_str(), m_para->serNo.c_str(), namesId, m_para->detSegment);
  // The raw block in both modes, typed u16: a flat byte array would misdescribe it
  RawU16Def dataDef;
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

  // Load the calibration constants onto the GPU.
  std::vector<float> peds(NRanges * NPixels, 0.0);
  std::vector<float> gains(NRanges * NPixels, 1.0);
  auto const serNo = m_para->kwargs.find("calibSerNo");
  if (serNo != m_para->kwargs.end() && !m_passthru) {   // Pass-through records raw, so a CALIB run can take the first darks
    auto calibErr = fetchEpixUHRconstants("pedestals", serNo->second, runInfo, EpixUHRgainModes, NRanges, peds, m_warnings);
    calibErr |= fetchEpixUHRconstants("gain_factors", serNo->second, runInfo, EpixUHRgainModes, NRanges, gains, m_warnings);
    if (calibErr) {
      m_error = "failed to fetch calibration constants";
      rc = 1;
    }
  }
  else if (!m_passthru) {
    m_warnings.push_back("no calibSerNo kwarg: using pedestals of 0 and gains of 1");
  }
  chkError(cudaMemcpy(m_pedsVec_d,  peds.data(),  peds.size()  * sizeof(*m_pedsVec_d),  cudaMemcpyDefault));
  chkError(cudaMemcpy(m_gainsVec_d, gains.data(), gains.size() * sizeof(*m_gainsVec_d), cudaMemcpyDefault));

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

// Copies the u16 pixels after the TimingHeader into the raw block, uncalibrated, for
// CALIB and for prescaling.  Test pyld.raw, never keepRaw: in pass-through honouring
// keepRaw would zero every unmarked frame.
static __device__
void _copyRaw(const EventPayload& pyld, unsigned tid, unsigned stride)
{
  if (!pyld.raw)  return;               // No raw block this event: nothing to fill

  auto const __restrict__ src = (uint16_t const*)(pyld.data + sizeof(Pds::TimingHeader));
  auto const __restrict__ dst = (uint16_t*)pyld.raw;
  auto const payloadCnt = (pyld.size - sizeof(Pds::TimingHeader))/sizeof(*src);
  auto const rawCnt     = pyld.rawCnt / sizeof(*dst);
  // The array offline sees is a fixed NPixels, so a short payload leaves zeros
  // rather than shrinking it.  Damage::MissingData in event() carries the fact.
  auto const nElem = payloadCnt > rawCnt ? rawCnt : payloadCnt;
  for (auto i = tid; i < rawCnt; i += stride)
    dst[i] = i < nElem ? src[i] : 0;
}

// CALIB mode: raw instead of calibrated, so nothing writes the calibrated buffer.
struct EpixUHRemuCalib
{
  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.hasData)  return;         // A transition, or nothing intelligible
    _copyRaw(pyld, tid, stride);
  }
};

// BEAM: calibrate every event, and also copy the raw data of the events the timing
// system marked (_copyRaw selects them)
struct EpixUHRemuPrescale
{
  PedGainCalib calib;

  __device__
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    calib.process(pyld, tid, stride);
    if (pyld.hasData)  _copyRaw(pyld, tid, stride);
  }
};

// Instantiating the kernel templates here puts the per-element work in the same CUDA
// module as the kernel, so it inlines.  See ReaderKernels.cuh.
void EpixUHRemu::recordEvent(cudaStream_t           stream,
                             unsigned               blocks,
                             unsigned               threads,
                             const EventKernelArgs& args)
{
  if (m_passthru) {
    _event<EpixUHRemuCalib><<<blocks, threads, 0, stream>>>(args, EpixUHRemuCalib{});
    return;
  }

  // No reference buffers: only the simulator controls the raw data it generates,
  // so only it can supply something to verify the calibration against
  PedGainCalib const calib{pedestals_d(),
                           gains_d(),
                           nullptr,
                           0,
                           rangeOffset(),
                           rangeBits(),
                           dataOffset(),
                           dataBits()};
  _event<EpixUHRemuPrescale><<<blocks, threads, 0, stream>>>(args, EpixUHRemuPrescale{calib});
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHRemu(para, pool);
}

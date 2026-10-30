#include "EpixUHR3x2.hh"

#include "BEBDetector.hh"
#include "PythonConfigScanner.hh"
#include "XpmInfo.hh"

#include "psalg/calib/NDArray.hh"
#include "psalg/detector/UtilsConfig.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/Semaphore.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/VarDef.hh"

#include <Python.h>

#include <assert.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>

#include <random> // For fake data generation.

using logging = psalg::SysLog;
using json = nlohmann::json;

//  Limit error messages per enable/disable cycle
static const unsigned NERROR_PRINTS = 20;

static Drp::EpixUHR3x2* epix = 0;
static struct sigaction old_actions[64];

static void sigHandler(int signal)
{
    psignal(signal, "epixUHR received signal");

    // Deliberately no monStreamEnable() here: it calls into CPython, and a
    // signal is delivered on whichever thread the kernel picks, which need not
    // hold the GIL -- so the call faults and the SIGSEGV buries the signal that
    // prompted it.  Re-enabling the monitor stream belongs on the orderly
    // shutdown path, where the GIL can be held; see the TODO item.

    sigaction(signal, &old_actions[signal], NULL);
    raise(signal);
}

namespace Drp {

  class EpixUHR3x2RawDef : public XtcData::VarDef
  {
  public:
    enum index { raw };

    EpixUHR3x2RawDef() {
      NameVec.push_back({"raw", XtcData::Name::UINT16, 2});
    }
  } epixUHR3x2RawDef;

  EpixUHR3x2::EpixUHR3x2(Parameters* para, MemPool* pool)
    : BEBDetector   (para, pool)
    , m_env_sem     (Pds::Semaphore::FULL)
    , m_env_empty   (true)
    , m_monStreamOn (true)      // The stream is on until Enable disables it
  {
    // VC 0 is for data. VC 1 loopback
    virtChan = 0;

    // m_descramble = true;

    epix = this;

    struct sigaction sa;
    sa.sa_handler = sigHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;

    // Don't call the BEBDetector version of _init. Call the overload so that selection
    // of GPU or CPU path can be made, as well as choice of emulator or not.
    // Those decisions must happen before CONFIGURE (at rogue startup)
    // Only ePixUHR currently uses this, so updating the BEBDetector function
    // or making it virtual seems less appropriate
    _init_data_path_or_emu();

#define REGISTER(t) {                                   \
      if (sigaction(t, &sa, &old_actions[t]) < 0)       \
        printf("Couldn't set up %s handler\n", #t);     \
    }

    REGISTER(SIGINT);
    REGISTER(SIGTERM);           // What Slurm sends to stop a task
    REGISTER(SIGQUIT);
    REGISTER(SIGABRT);
    REGISTER(SIGSEGV);           // SIGKILL cannot be caught, so don't try

#undef REGISTER
  }

  void EpixUHR3x2::_init_data_path_or_emu() {
    const std::string detname { m_para->detName };
    const std::string data_fpga { m_para->device };
    const size_t data_lane_mask { m_para->laneMask };
    const std::string detType { m_para->detType };

    const std::string module_name {"psdaq.configdb." + detType + "_config"};

    m_module = _check(PyImport_ImportModule(module_name.c_str()));

    PyObject* func_dict = _check(PyModule_GetDict(m_module));
    {
      std::string init_func_name {detType + "_init"};
      PyObject* init_func = _check(PyDict_GetItemString(func_dict,
                                                        const_cast<char*>(init_func_name.c_str())));
      const char* xpmpv = [&]() {
        if (m_para->kwargs.count("xpmpv")) {
          return m_para->kwargs["xpmpv"].c_str();
        }
        return "";
      }();
      const char* timebase = [&]() {
        if (m_para->kwargs.count("timebase")) {
          return m_para->kwargs["timebase"].c_str();
        }
        return "186M";
      }();

      const int use_gpu = [&]() {
        if (m_para->kwargs.count("useGpu")) {
          return std::stoi(m_para->kwargs["useGpu"]);
        }
        return 1;
      }();
      const int is_emulator = [&]() {
        if (m_para->kwargs.count("isEmulator")) {
          return std::stoi(m_para->kwargs["isEmulator"]);
        }
        return 0;
      }();

      // Argument string: "s si ssi ii"
      // - s   : for `arg` (Not sure what this is for, but leaving for similarity sake)
      // - si  : for dev and lanemask
      // - ssi : for xpmpv, timebase, verbosity
      // - ii  : for use_gpu and is_emulator
      m_root = _check(PyObject_CallFunction(init_func,
                                            "ssissiii",
                                            detname.c_str(),
                                            data_fpga.c_str(),
                                            data_lane_mask,
                                            xpmpv,
                                            timebase,
                                            m_para->verbose,
                                            use_gpu,
                                            is_emulator));

      logging::info("Configured data path for %s (Emulator status: %s).",
                    use_gpu ? "the GPU" : "the CPU",
                    is_emulator ? "IS Emulator" : "NOT Emulator");
      if (m_root) {
        PyObject* o_virtChan = PyDict_GetItemString(m_root, "virtChan");
        if (o_virtChan) {
          virtChan = PyLong_AsLong(o_virtChan);
        }
      }

      m_configScanner = new PythonConfigScanner(*m_para, *m_module);
    }
  }

  EpixUHR3x2::~EpixUHR3x2() {}

  nlohmann::json EpixUHR3x2::connectionInfo(const nlohmann::json& msg) {
    std::string alloc_json = msg.dump();

    char func_name[64];
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    sprintf(func_name,"%s_connectionInfo",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"Os",m_root,alloc_json.c_str()));

    m_paddr = PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr"));
    printf("*** EpixUHR3x2: paddr is %08x = %u\n", m_paddr, m_paddr);

    // there is currently a failure mode where the register reads
    // back as zero or 0xffffffff (incorrectly). This is not the best
    // longterm fix, but throw here to highlight the problem. the
    // difficulty is that Matt says this register has to work
    // so that an automated software solution would know which
    // xpm TxLink's to reset (a chicken-and-egg problem) - cpo
    // Also, register is corrupted when port number > 15 - Ric
    if (!m_paddr || m_paddr==0xffffffff || (m_paddr & 0xff) > 15) {
      logging::critical("XPM Remote link id register illegal value: 0x%x. Try XPM TxLink reset.",m_paddr);
      abort();
    }

    nlohmann::json fullMsg = xpmInfo(m_paddr);
    PyObject* pyHashedSerNo = PyDict_GetItemString(mbytes, "short_sn_id");
    if (pyHashedSerNo) {
      auto* pystr = _check(PyUnicode_AsASCIIString(pyHashedSerNo));
      fullMsg["short_sn_id"] = std::string(PyBytes_AsString(pystr));
      Py_DECREF(pystr);
    } else {
      logging::warning("No short SN found - serial number configdb will be disabled!");
    }
    m_para->serNo = _string_from_PyDict(mbytes, "serno");
    Py_DECREF(mbytes);

    return fullMsg;
  }

  void EpixUHR3x2::_connectionInfo(PyObject* mbytes)
  {
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
  }

  unsigned EpixUHR3x2::enable(XtcData::Xtc& xtc,
                              const void* bufEnd,
                              const nlohmann::json& info)
  {
    m_nprints = NERROR_PRINTS;
    logging::debug("EpixUHR3x2 enable");
    monStreamDisable();
    return 0;
  }

  unsigned EpixUHR3x2::disable(XtcData::Xtc& xtc,
                               const void* bufEnd,
                               const nlohmann::json& info)
  {
    logging::debug("EpixUHR3x2 disable");
    monStreamEnable();
    return 0;
  }

  unsigned EpixUHR3x2::_configure(XtcData::Xtc& xtc,
                                  const void* bufEnd,
                                  XtcData::ConfigIter& configo)
  {
    {
      Alg alg("raw", m_rawAlgVersion[0], m_rawAlgVersion[1], m_rawAlgVersion[2]);

      Names& configNames =
        configo.namesLookup()[NamesId(nodeId, ConfigNamesIndex+1)].names();

      NamesId nid = m_evtNamesId[0] = NamesId(nodeId, EventNamesIndex);

      logging::debug("Constructing panel eventNames src 0x%x",
                     unsigned(nid));
      Names& eventNames = *new(xtc, bufEnd) Names(bufEnd,
                                                  configNames.detName(),
                                                  alg,
                                                  configNames.detType(),
                                                  configNames.detId(),
                                                  nid,
                                                  m_para->detSegment);

      eventNames.add(xtc, bufEnd, epixUHR3x2RawDef);
      m_namesLookup[nid] = NameIndex(eventNames);

    }

    {

      XtcData::Names& names = detector::configNames(configo);
      XtcData::DescData& descdata = configo.desc_shape();

      for(unsigned i = 0; i < names.num(); ++i) {
        XtcData::Name& name = names.get(i);
        if (strcmp(name.name(),"user.asic_enable") == 0) {
          // Check which asics have been enabled. This is a bit mask with bit matching
          // to asic numbered beginning with 1. 0b111111 == 63 == [A1, A2, A3, A4, A5, A6]
          m_asics = descdata.get_value<uint32_t>(name.name());
        }
      }
    }
    return 0;
  }

  //
  //  The timing header is in each ASIC pair batch
  //

  Pds::TimingHeader* EpixUHR3x2::getTimingHeader(uint32_t index) const
  {
    EvtBatcherHeader* ebh = static_cast<EvtBatcherHeader*>(m_pool->dmaBuffers[index]);
    ebh = reinterpret_cast<EvtBatcherHeader*>(ebh->next());
    uint32_t* p = reinterpret_cast<uint32_t*>(ebh);
    return reinterpret_cast<Pds::TimingHeader*>(p);
  }

  /**
   * Build raw data stream into the XTC2 output.
   * Asic data is physically arranged as:
   *
   *     A1   |   A3   |   A5   |
   *  --------+--------+--------+
   *     A0   |   A2   |   A4   |
   *
   * The firmware, beginning with v5.0.0, does the necessary ASIC rearrangements
   * to produce a single "panel" output, with single header and timing information.
   * Any disabled ASICs are zero-filled by the firmware.
   *
   * The mapping is then:
   * - TDEST 0 <-> Trigger
   * - TDEST 1 <-> Event
   * - TDEST 2 <-> Timing
   * - TDEST 3 <-> Panel data (with zero-filled disabled ASICs)
   *
   * For posterity the old arrangement is left below this line in the comment.
   * ---------------------------------------------------------------------------
   *
   * Prior to v5.0.0, the data came in as:
   *
   * This is also how the data arrives:
   * - 3 streams on each unbatcher (two of them) -- If using bifurcation
   *   - Stream 0 has A1, A3, A5
   *   - Stream 1 has A0, A2, A4
   * - 6 streams if not using bifurcation
   *   - A1, A3, A5, A0, A2, A4
   *
   * There is also the following before the raw asic data:
   * - Trigger (XPM) on TDEST 0
   * - Event         on TDEST 1
   * - Timing        on TDEST 2
   *
   * So the final mapping is:
   * - A0 <-> TDEST 6
   * - A1 <-> TDEST 3
   * - A2 <-> TDEST 7
   * - A3 <-> TDEST 4
   * - A4 <-> TDEST 8
   * - A5 <-> TDEST 5
   *
   * Alternatively, in order of subframes:
   * - TDEST 0 <-> Trigger
   * - TDEST 1 <-> Event
   * - TDEST 2 <-> Timing
   * - TDEST 3 <-> Asic 1
   * - TDEST 4 <-> Asic 3
   * - TDEST 5 <-> Asic 5
   * - TDEST 6 <-> Asic 0
   * - TDEST 7 <-> Asic 2
   * - TDEST 8 <-> Asic 4
   *
   * NOTE: Importantly, if you maintain this ordering, the data will map naturally
   * to a contiguous C-ordered (Row-Major) array. This works naturally for conversion
   * to NumPy.
   *
   * Additionally, the ePixUHR can be operated in two modes:
   * - "Gain Expanded": In which the data are 16 bit, and really float16,
   *   having been calibrated.
   * - 12-bit packed: In which the data are 12 bit, for efficient transfer.
   *   In this case, 11 bits are data (bits 1-11) and bit 0 is the gain bit.
   *   -> NOTE: ** Importantly ** If the data are in the unexpanded format, they
   *           STILL arrive in 16 bits, so are not really "packed". To interpret
   *           the data, the bit arrangement, every 16 bits is:
   *
   *           G D D D D D D D D D D D U U U U
   *           | \___________________/ \_____/
   *          /            |              |
   *     Gain bit   11 bits of data  Unused bits
   *
   * @param[out] xtc The XTC to write into.
   * @param[in] bufEnd The end of the allocated buffer (used by xtcdata APIs).
   * @param[in] l1count The current L1Accept count.
   * @param[in] subframes The raw data subframes DMA'd in on this L1Accept.
   */
  void EpixUHR3x2::_event(XtcData::Xtc& xtc,
                          const void* bufEnd,
                          uint64_t l1count,
                          std::vector<XtcData::Array<uint8_t>>& subframes)
  {
    constexpr unsigned asicRows   { 168 };
    constexpr unsigned asicCols   { 192 };
    constexpr unsigned panelRows  { 2 * asicRows };  // 336
    constexpr unsigned panelCols  { 3 * asicCols };  // 576
    constexpr size_t   panelBytes { size_t(panelRows) * panelCols * sizeof(uint16_t) };
    constexpr unsigned panelTDEST { 3 };

    unsigned shape[MaxRank] { panelRows, panelCols };

    CreateData cd(xtc, bufEnd, m_namesLookup, m_evtNamesId[0]);
    logging::debug("Writing panel event src 0x%x", unsigned(m_evtNamesId[0]));

    Array<uint16_t> aframe = cd.allocate<uint16_t>(EpixUHR3x2RawDef::raw, shape);
    uint8_t* dest { reinterpret_cast<uint8_t*>(aframe.data()) };

    // Zero the data in case of missing asics -- Presumably done by firmware
    // std::memset(dest, 0, numAsicBytes * numAsics);

    if (panelTDEST >= subframes.size()) {
      xtc.damage.increase(XtcData::Damage::MissingData);
      std::memset(dest, 0, panelBytes);
      if (m_nprints) {
        --m_nprints;
        logging::error("ePixUHR3x2: No panel subframe (nsub %zu) L1Count %lu",
                       subframes.size(), l1count);
      }

      return;
    }

    auto& data_subframe { subframes[panelTDEST] };
    const auto*  src { reinterpret_cast<const uint8_t*>(data_subframe.data()) };
    const size_t nbytes { data_subframe.num_elem() };
    const size_t ncopy { std::min(nbytes, panelBytes) };

    std::memcpy(dest, src, ncopy);

    if (nbytes != panelBytes) {
      xtc.damage.increase(XtcData::Damage::MissingData);
      if (ncopy < panelBytes) {
        std::memset(dest + ncopy, 0, panelBytes - ncopy);
      }
      if (m_nprints) {
        --m_nprints;
        logging::error("ePixUHR3x2: Panel size is %zu, expected %zu",
                       nbytes, panelBytes);
      }
    }
  }

  void EpixUHR3x2::slowupdate(XtcData::Xtc& xtc, const void* bufEnd)
  {
    this->Detector::slowupdate(xtc, bufEnd);
  }

  bool EpixUHR3x2::scanEnabled()
  {
    //  Only needed this when the TimingSystem could not be used
    //    return true;
    return false;
  }

  void EpixUHR3x2::shutdown()
  {
    // Re-enable the monitor stream here rather than from sigHandler(), which
    // may not call CPython: the caller, _disconnect(), holds the GIL.  Not
    // done in the dtor, which has no such guarantee.
    if (m_module && !m_monStreamOn) {
      monStreamEnable();
      m_monStreamOn = true;
    }
  }

  void EpixUHR3x2::monStreamEnable()
  {
    m_monStreamOn = true;
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_disable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict,
                                                  reinterpret_cast<char*>(func_name)));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
  }

  void EpixUHR3x2::monStreamDisable()
  {
    m_monStreamOn = false;
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    char func_name[64];
    sprintf(func_name,"%s_enable",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict,
                                                  reinterpret_cast<char*>(func_name)));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "O", m_root));
    Py_DECREF(mybytes);
  }
} // namespace Drp


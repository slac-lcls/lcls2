#include "ReducerAlgo.hh"
#include "Detector.hh"
#include "MemPool.hh"

#include "drp/drp.hh"
#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/VarDef.hh"

#include "DrpAlgos/drp/KernelRegistry.hh"

#include <Python.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

using logging = psalg::SysLog;
using json = nlohmann::json;
using namespace XtcData;
using namespace Drp::Gpu;

namespace {
  /**
   * Runs the Reducer handshake for the DRP states.
   *
   * The algorithms themselves do not call this right now. Its inserted into
   * the graph for them.
   */
  __global__ void _drpFinalize(unsigned* const state,
                               const unsigned* const index,
                               uint8_t* const dataBuffers,
                               size_t const dataBufsCnt,
                               size_t const fixedSize) {
    if (*state != 1) {
      return;
    }

    if (fixedSize) {
      reinterpret_cast<size_t*>(&dataBuffers[*index * dataBufsCnt])[-1] = fixedSize;
    }

    *state = 2; // Hand off to _reducerLoop
  }

  // Temporary config fetching
  json _fetchDetConfig(const json& connectMsg,
                       const std::string& cfgtype,
                       const std::string& detName,
                       unsigned detSegment) {
    PyGILState_STATE gil { PyGILState_Ensure() };

    PyObject* mod = PyImport_ImportModule("psdaq.configdb.get_config");
    PyObject* res = mod
      ? PyObject_CallMethod(mod,
                            "get_config_json",
                            "sssi",
                            connectMsg.dump().c_str(),
                            cfgtype.c_str(),
                            detName.c_str(),
                            static_cast<int>(detSegment))
      : nullptr;

    json cfg;
    std::string err;
    if (res) {
      cfg = json::parse(PyUnicode_AsUTF8(res));
    } else {
      PyObject *type, *value, *tb;
      PyErr_Fetch(&type, &value, &tb);
      PyObject* s { value ? PyObject_Str(value) : nullptr };
      err = s ? PyUnicode_AsUTF8(s) : "unknown Python error";
      Py_XDECREF(s); Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(tb);
    }

    Py_XDECREF(res);
    Py_XDECREF(mod);
    PyGILState_Release(gil);

    if (!err.empty()) {
      throw std::runtime_error("get_config failed: " + err);
    }

    return cfg;
  }

  constexpr size_t kPayloadPad { 4096 };

  /// ---------------------- HACK ---------------------- ///
  /// This is a hack for the lack of a real CONFIGURE included with the
  /// fakecam-based detectors. We dump the parameters into a special section
  /// called drpalg

  // This index value for the XTC2 should be safe
  // It's likely that this stuff will be removed altogether eventually anyway
  constexpr unsigned ReducerConfigNamesIndex { unsigned(ReducerNamesIndex) + 1 };
  /// -------------------- END HACK -------------------- ///

} // anonymous namespace

namespace Drp::Gpu {
  class DrpAlgosReducer : public ReducerAlgo {
  public:
    DrpAlgosReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det)
      : ReducerAlgo(para, pool, det)
      , m_capacity(pool.calibBufsSize() + kPayloadPad)
    {}

    bool hasGraph() const override { return true; }

    size_t payloadSize() const override { return m_capacity; }

    int configure(const json& configureMsg, const json& connectMsg, size_t collectionId) override {
      // Using a temporary setup here which is inefficient. It refetches the config
      // since the configureMsg doesn't contain it. The alternative would be to do
      // the traversal and parsing through the pre-constrcuted XTC.
      // Can improve this over time as code is merged.
      json detCfg;
      try {
        detCfg = _fetchDetConfig(connectMsg,
                                 configureMsg.at("config_alias").get<std::string>(),
                                 m_para.detName,
                                 m_para.detSegment);
      } catch (const std::exception& e) {
        logging::error("DrpAlgosReducer: %s", e.what());
        return -1;
      }

      if (!detCfg.contains("drp_alg0")) {
        logging::error("DrpAlgosReducer: no drp_alg0 in configuration");
        return -1;
      }

      const json& alg { detCfg["drp_alg0"] };
      try {
        m_name = alg.at("name").get<std::string>();
        m_version = DrpAlgos::Version::parse(alg.at("version").get<std::string>());
        const auto soname = alg.at("soname").get<std::string>();

        m_desc = DrpAlgos::KernelRegistry::instance().create(m_name, m_version, soname);
        m_desc->setWorkers(m_para.nworkers);
        m_desc->configure(alg.at("parameters"));
      } catch (const std::exception& e) {
        logging::error("DrpAlgosReducer: %s", e.what());
        return -1;
      }

      /// ---------------------- HACK ---------------------- ///
      /// This is a hack for the lack of a real CONFIGURE included with the
      /// fakecam-based detectors. We dump the parameters into a special section
      /// called drpalg
      m_paramsJson = alg.at("parameters").dump();
      /// -------------------- END HACK -------------------- ///

      m_payloadSize = m_desc->payloadSize();

      // Reduce buffers are allocated in the Reducer constructor, before this runs.
      if (m_payloadSize > m_pool.reduceBufsSize()) {
        logging::error("DrpAlgosReducer: %s %s output (%zu B) exceeds reduce buffer (%zu B)",
                       m_name.c_str(),
                       m_version.toString().c_str(),
                       m_payloadSize,
                       m_pool.reduceBufsSize());
        return -1;
      }

      logging::info("DrpAlgosReducer: configured %s %s, payload %zu B",
                    m_name.c_str(),
                    m_version.toString().c_str(),
                    m_payloadSize);
      return 0;
    }

    void recordGraph(cudaStream_t stream,
                     unsigned* const state,
                     unsigned* const index,
                     float const* const calibBuffers,
                     size_t const calibBufsCnt,
                     uint8_t* const dataBuffers,
                     size_t const dataBufsCnt) override {
      // Construct the context
      // These are device pointers and read at run time
      DrpKernelCtx ctx {
        state,
        index,
        calibBuffers,
        calibBufsCnt,
        dataBuffers,
        dataBufsCnt,
        nullptr
      };

      try {
        m_desc->launch(ctx, stream);
      } catch (const std::exception& e) {
        logging::error("DrpAlgosReducer: %s launch failed during capture: %s",
                       m_name.c_str(),
                       e.what());

        // Forcing a sync will cause the graph capture to abort as its illegal
        cudaStreamSynchronize(stream);
        return;
      }

      _drpFinalize<<<1, 1, 0, stream>>>(state,
                                        index,
                                        dataBuffers,
                                        dataBufsCnt,
                                        m_desc->variableSize() ? 0 : m_payloadSize);

      cudaError_t err { cudaGetLastError() };
      if (err != cudaSuccess) {
        logging::error("DrpAlgosReducer: kernel launch failed during capture: %s",
                       cudaGetErrorString(err));
      }
    }

    void reduce(cudaGraphExec_t,
                cudaStream_t,
                unsigned,
                size_t* dataSize,
                unsigned* retCode) override {
      // This is only called with HOST_LAUNCHED_REDUCERS
      // DRP-algos doesn't currently implement support (can be added fairly easily)
      logging::error("DrpAlgosReducer: host-launched reducers are not supported");

      *dataSize = 0;
      *retCode = 1;
    }

    unsigned configure(Xtc& xtc, const void* bufEnd) override {
      // XTC2 Alg is given the name of the DRP reduction algorithm and its version
      Alg alg(m_name.c_str(), m_version.major, m_version.minor, m_version.micro);
      NamesId namesId(m_det.nodeId, ReducerNamesIndex);
      Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                             m_para.detName.c_str(), alg,
                                             m_para.detType.c_str(), m_para.serNo.c_str(),
                                             namesId, m_para.detSegment);

      VarDef varDef;
      varDef.NameVec.push_back({ "reduced", Name::UINT8, 1 });
      names.add(xtc, bufEnd, varDef);
      m_det.namesLookup()[namesId] = NameIndex(names);

      /// ---------------------- HACK ---------------------- ///
      /// This is a hack for the lack of a real CONFIGURE included with the
      /// fakecam-based detectors. We dump the parameters into a special section
      /// called drpalg
      Alg cfgAlg("drpalg", 0, 1, 0);
      NamesId cfgNamesId(m_det.nodeId, ReducerConfigNamesIndex);
      Names& cfgNames =
      *new (xtc, bufEnd) Names(bufEnd,
                               m_para.detName.c_str(),
                               cfgAlg,
                               m_para.detType.c_str(),
                               m_para.serNo.c_str(),
                               cfgNamesId,
                               m_para.detSegment);

      VarDef cfgDef;
      cfgDef.NameVec.push_back({ "algname", Name::CHARSTR, 1 });
      cfgDef.NameVec.push_back({ "algversion", Name::CHARSTR, 1 });
      cfgDef.NameVec.push_back({ "parameters", Name::CHARSTR, 1 });
      cfgNames.add(xtc, bufEnd, cfgDef);
      m_det.namesLookup()[cfgNamesId] = NameIndex(cfgNames);

      CreateData cfg(xtc, bufEnd, m_det.namesLookup(), cfgNamesId);
      cfg.set_string(0, m_name.c_str());
      cfg.set_string(1, m_version.toString().c_str());
      cfg.set_string(2, m_paramsJson.c_str());
      /// -------------------- END HACK -------------------- ///

      return 0;
    }

    void event(Xtc& xtc, const void* bufEnd, unsigned dataSize) override {
      NamesId namesId(m_det.nodeId, ReducerNamesIndex);
      CreateData data(xtc, bufEnd, m_det.namesLookup(), namesId);
      unsigned dataShape[MaxRank] = { dataSize };
      data.set_array_shape(0u, dataShape);
    }

  private:
    std::string m_name;
    DrpAlgos::Version m_version;
    std::shared_ptr<DrpAlgos::IKernelDescriptor> m_desc;
    size_t m_payloadSize { 0 };
    size_t m_capacity { 0 };

    /// ---------------------- HACK ---------------------- ///
    /// This is a hack for the lack of a real CONFIGURE included with the
    /// fakecam-based detectors. We dump the parameters into a special section
    /// called drpalg
    std::string m_paramsJson {};
    /// -------------------- END HACK -------------------- ///
  };
} // namespace Drp::Gpu

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters& para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector& det)
{
  return new Drp::Gpu::DrpAlgosReducer(para, pool, det);
}

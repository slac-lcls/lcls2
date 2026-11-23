#include "ReducerAlgo.hh"
#include "Detector.hh"
#include "MemPool.hh"

#include "drp/drp.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "psdaq/service/MetricExporter.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/VarDef.hh"

#include "DrpAlgos/drp/BuildInfo.hh"
#include "DrpAlgos/drp/KernelRegistry.hh"
#include "DrpAlgos/drp/Launch.hh"
#include "DrpAlgos/drp/Metrics.hh"
#include "DrpAlgos/drp/Requirements.hh"

#include <Python.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using logging = psalg::SysLog;
using json = nlohmann::json;
using namespace XtcData;
using namespace Drp::Gpu;


namespace {
  /// The DRP algorithm configuration document (Json2Xtc-ready), or "" if none
  std::string _fetchDrpAlgConfig(const json& connectMsg,
                                 const std::string& cfgtype,
                                 const std::string& detName,
                                 unsigned detSegment) {
    PyGILState_STATE gil { PyGILState_Ensure() };

    PyObject* mod { PyImport_ImportModule("psdaq.configdb.get_config") };
    PyObject* res {
      mod ? PyObject_CallMethod(mod,
                                "get_drp_alg_config_json",
                                "sssi",
                                connectMsg.dump().c_str(),
                                cfgtype.c_str(),
                                detName.c_str(),
                                static_cast<int>(detSegment))
          : nullptr
    };

    std::string doc;
    std::string err;
    if (res) {
      const char* s { PyUnicode_AsUTF8(res) };
      doc = s ? s : "";
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
      throw std::runtime_error("get_drp_alg_config failed: " + err);
    }

    return doc;
  }

  constexpr size_t kPayloadPad { 4096 };
} // anonymous namespace

namespace Drp::Gpu {
  class DrpAlgosReducer : public ReducerAlgo {
  public:
    DrpAlgosReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det)
      : ReducerAlgo(para, pool, det)
      , m_capacity(pool.calibBufsSize() + kPayloadPad)
    {}
    ~DrpAlgosReducer() override { _freeSeq(); }

    size_t payloadSize() const override { return m_capacity; }

    int configure(const json& configureMsg, const json& connectMsg, size_t collectionId) override {
      m_wantsVetoed = false; // Reset in case of failure below

      json alg;
      std::string params;
      try {
        const std::string fetched = _fetchDrpAlgConfig(connectMsg,
                                                       configureMsg.at("config_alias").get<std::string>(),
                                                       m_para.detName,
                                                       m_para.detSegment);

        if (fetched.empty()) {
          logging::error("DrpAlgosReducer: no drp_alg0 in configuration");
          return -1;
        }
        const json doc = json::parse(fetched); // Wrong constructor used if brace-intialized!
        // We return an "XTC-ready" version of the parameters under key `xtc`
        // - This includes flattened arrays etc.
        // Under `parameters`, the unmodified config is stored - easier for the reducer
        // In addition, we add the actual build and NVRTC info into the written output.
        int driver { 0 };
        cudaDriverGetVersion(&driver);
        json algCfgXtc = doc.at("xtc"); // DONT USE BRACE-INIT!

        // NOTE: This will need to be updated if/when more than 1 DRP algo is allowed
        //       (they will then be under, `drp_alg0`, `drp_alg1`, etc.)
        algCfgXtc["drp_alg0"]["gitBuildVer"]  = DrpAlgos::gitBuildVersion();
        algCfgXtc["drp_alg0"]["jitNvrtcVer"]  = DrpAlgos::jitNvrtcVersion();
        algCfgXtc["drp_alg0"]["cudaDriverVer"] = driver;
        algCfgXtc[":types:"]["drp_alg0"]["gitBuildVer"]  = "CHARSTR";
        algCfgXtc[":types:"]["drp_alg0"]["jitNvrtcVer"]  = "CHARSTR";
        algCfgXtc[":types:"]["drp_alg0"]["cudaDriverVer"] = "INT32";
        m_algConfig = algCfgXtc.dump();
        alg = doc.at("xtc").at("drp_alg0");
        params = doc.at("parameters").dump();
      } catch (const std::exception& e) {
        logging::error("DrpAlgosReducer: %s", e.what());
        return -1;
      }

      try {
        m_name = alg.at("name").get<std::string>();
        m_version = DrpAlgos::Version::parse(alg.at("version").get<std::string>());
        const auto soname { alg.at("soname").get<std::string>() };

        m_desc = DrpAlgos::KernelRegistry::instance().create(m_name, m_version, soname);
        m_desc->setWorkers(m_para.nworkers);
        m_desc->configure(params.c_str(), params.size());

        constexpr std::uint32_t kSupported { DrpAlgos::Requirements_v1::kVetoedEvents };
        const auto* req {
          static_cast<const DrpAlgos::Requirements_v1*>(m_desc->query(DrpAlgos::Requirements_v1::id))
        };
        const std::uint32_t required { req ? req->required : 0u };
        if (required & ~kSupported) {
          logging::error("DrpAlgosReducer: %s %s requires unsupported features 0x%x",
                         m_name.c_str(),
                         m_version.toString().c_str(),
                         required & ~kSupported);
          return -1;
        }
        m_wantsVetoed = required & DrpAlgos::Requirements_v1::kVetoedEvents;
      } catch (const std::exception& e) {
        logging::error("DrpAlgosReducer: %s", e.what());
        return -1;
      }

      m_payloadSize = m_desc->payloadSize();
      if (m_payloadSize > m_pool.reduceBufsSize()) {
        logging::error("DrpAlgosReducer: %s %s output (%zu B) exceeds reduce buffer (%zu B)",
                       m_name.c_str(), m_version.toString().c_str(),
                       m_payloadSize, m_pool.reduceBufsSize());
        return -1;
      }

      // Setup the sequence info, for round-robin, per-worker event sequence
      // has worker w's first event as w
      _freeSeq();
      m_seq_d.resize(m_para.nworkers, nullptr);
      for (unsigned w = 0; w < m_para.nworkers; ++w) {
        const unsigned long long first { w };
        if (cudaMalloc(&m_seq_d[w], sizeof(first)) != cudaSuccess ||
            cudaMemcpy(m_seq_d[w], &first, sizeof(first), cudaMemcpyHostToDevice) != cudaSuccess) {
          logging::error("DrpAlgosReducer: failed to set up seq for worker %u", w);
          return -1;
        }
      }
      m_nextWorker = 0;

      logging::info("DrpAlgosReducer: configured %s %s (ctx ABI %u), payload %zu B",
                    m_name.c_str(), m_version.toString().c_str(),
                    m_desc->ctxAbi(), m_payloadSize);
      return 0;
    }

    void recordGraph(cudaStream_t stream,
                     unsigned* const state,
                     unsigned* const index,
                     float const* const calibBuffers,
                     size_t const calibBufsCnt,
                     uint8_t* const dataBuffers,
                     size_t const dataBufsCnt,
                     unsigned* const retCode) override {
      const unsigned worker { m_nextWorker++ };
      if (worker >= m_seq_d.size()) {
        logging::error("DrpAlgosReducer: more graphs recorded (%u) than workers (%zu)",
                       worker + 1, m_seq_d.size());
        // Synchronizing a stream that is being captured invalidates that process
        // cudaStreamEndCapture fails, and the configure fails as a result
        cudaStreamSynchronize(stream);
        return;
      }

      // ALWAYS build the latest ABI. launchAnyAbi converts for older descriptors
      const DrpAlgos::LatestCtx ctx {
        .abi       = DrpAlgos::LatestCtx::kABI,
        .gate      = state,
        .slot      = index,
        .in        = calibBuffers,
        .inStride  = calibBufsCnt,
        .out       = dataBuffers,
        .outStride = dataBufsCnt,
        .retCode   = retCode,
        .seq       = m_seq_d[worker],
        .rawBytes  = 0,              // Only pass-through puts raw in the reduce buffers
        .params    = nullptr,
      };

      try {
        DrpAlgos::recordAlgorithm(*m_desc, ctx, m_para.nworkers, stream);
      } catch (const std::exception& e) {
        logging::error("DrpAlgosReducer: %s launch failed during capture: %s",
                       m_name.c_str(), e.what());
        // Synchronizing a stream that is being captured invalidates that process
        // cudaStreamEndCapture fails, and the configure fails as a result
        cudaStreamSynchronize(stream);
        return;
      }
    }

    unsigned configure(Xtc& xtc, const void* bufEnd) override {
      // XTC2 Alg is given the name of the DRP reduction algorithm and its version
      // (reduced data)
      Alg alg(m_name.c_str(), m_version.major, m_version.minor, m_version.micro);
      NamesId namesId(m_det.nodeId, ReducerNamesIndex);
      Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                             m_para.detName.c_str(),
                                             alg,
                                             m_para.detType.c_str(),
                                             m_para.serNo.c_str(),
                                             namesId,
                                             m_para.detSegment);
      VarDef varDef;
      varDef.NameVec.push_back({ "reduced", Name::UINT8, 1 });
      names.add(xtc, bufEnd, varDef);
      m_det.namesLookup()[namesId] = NameIndex(names);

      // Algorithm configuration, through the standard Json2Xtc translation
      std::vector<char> buf(m_para.maxTrSize);
      const int len {
        Pds::translateJson2Xtc(m_algConfig.data(),
                               buf.data(),
                               buf.data() + buf.size(),
                               NamesId(m_det.nodeId, ReducerConfigNamesIndex),
                               m_para.detName.c_str(),
                               m_para.detSegment)
      };
      if (len <= 0) {
        logging::error("DrpAlgosReducer: Json2Xtc failed for the %s configuration",
                       m_name.c_str());
        return 1;
      }

      const Xtc& parent { *reinterpret_cast<const Xtc*>(buf.data()) };
      void* payload { xtc.alloc(parent.sizeofPayload(), bufEnd) };
      std::memcpy(payload, parent.payload(), parent.sizeofPayload());

      return 0;
    }

    void event(Xtc& xtc, const void* bufEnd, unsigned dataSize) override {
      NamesId namesId(m_det.nodeId, ReducerNamesIndex);
      CreateData data(xtc, bufEnd, m_det.namesLookup(), namesId);
      unsigned dataShape[MaxRank] = { dataSize };
      data.set_array_shape(0u, dataShape);
    }

    int setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter,
                     std::map<std::string, std::string>& labels) override {
      if (!m_desc) {
        return 0;
      }

      auto* const iface {
        static_cast<const DrpAlgos::IMetrics_v1*>(m_desc->query(DrpAlgos::IMetrics_v1::id))
      };
      if (!iface) {
        // If nullptr, the reducer algorithm reports no metrics (at the request capability)
        return 0;
      }

      const DrpAlgos::IMetrics_v1::Metric* const list { iface->metrics() };
      for (std::size_t i = 0; i < iface->metricCount(); ++i) {
        // All v1 metrics have a name, kind, and device side value.
        // See DrpAlgos/drp/Metrics.hh for impl.
        const auto& metric { list[i] };
        const unsigned long long* const dev { metric.dev };
        // The integer kind is 0 for COUNTER, 1 for GAUGE (on v1 capable implementations)
        const Pds::MetricType type {
          metric.kind == 0 ? Pds::MetricType::Counter : Pds::MetricType::Gauge
        };

        auto copy_metric = [this, dev]() -> int64_t {
          // The exporter's threads have no CUDA context
          if (cuCtxSetCurrent(m_pool.context().context()) != CUDA_SUCCESS) {
            return 0;
          }

          unsigned long long value { 0 };
          if (cudaMemcpy(&value, dev, sizeof(value), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return 0;
          }

          return static_cast<int64_t>(value);
        };

        // Prepand all names with DRP_
        exporter->add(std::string("DRP_") + metric.name, labels, type, copy_metric);
      }

      logging::info("DrpAlgosReducer: %s exports %zu metrics", m_name.c_str(), iface->metricCount());
      return 0;
    }

    bool wantsVetoed() const override { return m_wantsVetoed; }

  private:
    void _freeSeq() {
      for (auto* p : m_seq_d) {
        if (p) {
          cudaFree(p);
        }
      }
      m_seq_d.clear();
    }

    std::string m_name;
    DrpAlgos::Version m_version;
    std::shared_ptr<DrpAlgos::IKernelDescriptor> m_desc;
    size_t m_payloadSize { 0 };
    size_t m_capacity { 0 };
    std::string m_algConfig;                    ///< Json2Xtc-ready configuration document
    std::vector<unsigned long long*> m_seq_d;   ///< Per-worker event sequence (device)
    unsigned m_nextWorker { 0 };                ///< recordGraph is called once per worker, in order
    bool m_wantsVetoed { false };               ///< Whether the algorithm requires dispatch of vetoed L1Accepts (for book-keeping)
  };
} // namespace Drp::Gpu

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters& para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector& det)
{
  return new Drp::Gpu::DrpAlgosReducer(para, pool, det);
}

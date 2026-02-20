#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <Python.h>
#include "DrpBase.hh"
#include "XpmDetector.hh"
#include "spscqueue.hh"
#include "psdaq/epicstools/PvMonitorBase.hh"
#include "psdaq/service/Collection.hh"

namespace Drp {

struct PvParameters;
class  PvDetector;

class Pgp : public PgpReader
{
public:
    Pgp(const Parameters& para, MemPool& pool, Detector* det);
    Pds::EbDgram* next(uint32_t& evtIndex);
    const uint64_t nDmaRet() const { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    Detector* m_det;
    static const int MAX_RET_CNT_C = 100;
    int32_t m_available;
    int32_t m_current;
    uint64_t m_nDmaRet;
};


class PvMonitor : public Pds_Epics::PvMonitorBase
{
public:
    PvMonitor(const PvParameters&      para,
              const std::string&       alias,
              const std::string&       pvName,
              const std::string&       provider,
              const std::string&       request,
              const std::string&       field,
              unsigned                 id,
              size_t                   nBuffers,
              size_t                   bufferSize,
              unsigned                 type,
              size_t                   nelem,
              size_t                   rank,
              uint32_t                 firstDim,
              const std::atomic<bool>& running,
              PvDetector&              pvDetector);
public:
    void onConnect()    override;
    void onDisconnect() override;
    void updated()      override;
public:
    void startup();
    void shutdown();
    int  getParams(std::string& fieldName, XtcData::Name::DataType& xtcType, int& rank);
    unsigned id() const { return m_id; }
    const std::string& alias() const { return m_alias; }
    uint64_t nUpdates() const { return m_nUpdates; }
    uint64_t nMatch()   const { return m_nMatch; }
    uint64_t nEmpty()   const { return m_nEmpty; }
    uint64_t nTooOld()  const { return m_nTooOld; }
    int64_t  timeDiff() const { return m_timeDiff; }
    int64_t  latency()  const { return m_latency; }
private:
    void _tL1EqPv(Pds::EbDgram*, const XtcData::TimeStamp&);
    void _tL1LtPv(Pds::EbDgram*, const XtcData::TimeStamp&);
    void _tL1GtPv(Pds::EbDgram*, const XtcData::TimeStamp&);
private:
  enum State { NotReady, Armed, Ready };
private:
    const Parameters&               m_para;
    mutable std::mutex              m_mutex;
    mutable std::condition_variable m_condition;
    State                           m_state;
    unsigned                        m_id;
    unsigned                        m_type;
    size_t                          m_nelem;
    size_t                          m_rank;
    size_t                          m_payloadSize;
    size_t                          m_bufferSize;
    uint32_t                        m_firstDimOverride;
    std::string                     m_alias;
    const std::atomic<bool>&        m_running;
public:
    SPSCQueue<Pds::EbDgram*>        pvQueue;
    SPSCQueue<Pds::EbDgram*>        l1Queue;
private:
    PvDetector&                     m_det;
    ZmqContext                      m_context;
    ZmqSocket                       m_notifySocket;
    uint64_t                        m_nUpdates;
    uint64_t                        m_nMatch;
    uint64_t                        m_nEmpty;
    uint64_t                        m_nTooOld;
    int64_t                         m_timeDiff;
    int64_t                         m_latency;
};


class PvDetector : public XpmDetector
{
public:
    PvDetector(PvParameters&, MemPoolCpu&);
    ~PvDetector();
    unsigned connect(const nlohmann::json&, const std::string& collectionId, std::string& msg);
    unsigned disconnect();
    unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
    unsigned unconfigure();
    using Detector::enable;             // Avoid 'hidden' warning
    void enable();
    using Detector::disable;            // Avoid 'hidden' warning
    void disable();
    void event(XtcData::Dgram& evt, const void* bufEnd, PGPEvent*, uint64_t l1count) override { /* unused */ };
    void event(XtcData::Dgram& evt, const void* bufEnd, const Pds::Eb::ResultDgram&) override { /* unused */ };
    const std::vector< std::shared_ptr<PvMonitor> >& pvMonitors() const { return m_pvMonitors; }
public:
    static const unsigned maxSupportedPVs = 64;
    enum {
      ConfigNamesIndex = NamesIndex::BASE,
      RawNamesIndex = unsigned(ConfigNamesIndex) + maxSupportedPVs,
      InfoNamesIndex = unsigned(RawNamesIndex) + maxSupportedPVs,
    };
private:
    std::atomic<bool>                         m_running;
    std::vector< std::shared_ptr<PvMonitor> > m_pvMonitors;
    PyObject*                                 m_pyModule;
    std::string                               m_connectJson;
};


class PvDrp : public DrpBase
{
public:
    PvDrp(PvParameters&, MemPoolCpu&, PvDetector&, ZmqContext&);
    virtual ~PvDrp() {}
    std::string configure(const nlohmann::json& msg);
    unsigned unconfigure();
private:
    int  _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
    void _reader();
    void _collector();
    void _handleTransition(Pds::EbDgram& evtDg, Pds::EbDgram& trDg);
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    const PvParameters&      m_para;
    PvDetector&              m_det;
    Pgp                      m_pgp;
    std::thread              m_readerThread;
    std::thread              m_collectorThread;
    SPSCQueue<Pds::EbDgram*> m_evtQueue;
    std::atomic<bool>        m_terminate;
    uint64_t                 m_nEvents;
    uint64_t                 m_nUpdates;
    uint64_t                 m_nMatch;
    uint64_t                 m_nEmpty;
    uint64_t                 m_nTooOld;
    int64_t                  m_timeDiff;
};


class PvApp : public CollectionApp
{
public:
    PvApp(PvParameters& para);
    ~PvApp();
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connectionShutdown() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);
private:
    PvParameters&               m_para;
    MemPoolCpu                  m_pool;
    std::unique_ptr<PvDetector> m_det;
    std::unique_ptr<PvDrp>      m_drp;
    bool                        m_unconfigure;
};

}

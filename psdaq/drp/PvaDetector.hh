#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <mutex>
#include <chrono>
#include <condition_variable>
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
    Pgp(const Parameters& para, DrpBase& drp, Detector* det, const bool& running);
    Pds::EbDgram* next(uint32_t& evtIndex);
    const uint64_t nDmaRet() { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    Detector* m_det;
    Pds::Eb::TebContributor& m_tebContributor;
    static const int MAX_RET_CNT_C = 100;
    const bool& m_running;
    int32_t m_available;
    int32_t m_current;
    unsigned m_nodeId;
    uint64_t m_nDmaRet;
};


class PvMonitor : public Pds_Epics::PvMonitorBase
{
public:
    PvMonitor(const PvParameters& para,
              const std::string&  alias,
              const std::string&  pvName,
              const std::string&  provider,
              const std::string&  request,
              const std::string&  field,
              unsigned            id,
              size_t              nBuffers,
              uint32_t            firstDim);
public:
    void onConnect()    override;
    void onDisconnect() override;
    void updated()      override;
public:
    void startup();
    void shutdown();
    void timeout(const PgpReader& pgp, std::chrono::milliseconds timeout);
    int  getParams(std::string& fieldName, XtcData::Name::DataType& xtcType, int& rank);
    unsigned id() const { return m_id; }
    const std::string& alias() const { return m_alias; }
    uint64_t nUpdates() const { return m_nUpdates; }
    uint64_t nMissed() const { return m_nMissed; }
private:
    enum State { NotReady, Armed, Ready };
private:
    const Parameters&               m_para;
    mutable std::mutex              m_mutex;
    mutable std::condition_variable m_condition;
    pvd::ScalarType                 m_type;
    size_t                          m_nelem;
    size_t                          m_rank;
    size_t                          m_payloadSize;
    State                           m_state;
    unsigned                        m_id;
    uint32_t                        m_firstDimOverride;
    std::string                     m_alias;
public:
    SPSCQueue<XtcData::Dgram*>      pvQueue;
    SPSCQueue<XtcData::Dgram*>      bufferFreelist;
private:
    std::vector<uint8_t>            m_buffer;
    ZmqContext                      m_context;
    ZmqSocket                       m_notifySocket;
    uint64_t                        m_nUpdates;
    uint64_t                        m_nMissed;
};


class PvDetector : public XpmDetector
{
public:
    PvDetector(PvParameters& para, DrpBase& drp);
    unsigned connect(std::string& msg);
    unsigned disconnect();
  //    std::string sconfigure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd);
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override { /* unused */ };
    void event(XtcData::Dgram& dgram, const void* bufEnd, const XtcData::Xtc& pvXtc);
    unsigned unconfigure();
    const PgpReader* pgp() { return &m_pgp; }
private:
    void _worker();
    void _timeout(std::chrono::milliseconds timeout);
    void _matchUp();
    void _handleTransition(Pds::EbDgram& evtDg, Pds::EbDgram& trDg);
    void _tEvtEqPv(std::shared_ptr<PvMonitor>&, Pds::EbDgram& evtDg, const XtcData::Dgram& pvDg);
    void _tEvtLtPv(std::shared_ptr<PvMonitor>&, Pds::EbDgram& evtDg, const XtcData::Dgram& pvDg);
    void _tEvtGtPv(std::shared_ptr<PvMonitor>&, Pds::EbDgram& evtDg, const XtcData::Dgram& pvDg);
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    struct Event
    {
      uint32_t index;
      uint32_t remaining;
    };
private:
    enum {RawNamesIndex = NamesIndex::BASE, InfoNamesIndex};
    PvParameters& m_para;
    DrpBase& m_drp;
    Pgp m_pgp;
    std::vector< std::shared_ptr<PvMonitor> > m_pvMonitors;
    std::thread m_workerThread;
    SPSCQueue<Event> m_evtQueue;
    std::atomic<bool> m_terminate;
    std::atomic<bool> m_running;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    uint64_t m_nEvents;
    uint64_t m_nUpdates;
    uint64_t m_nMissed;
    uint64_t m_nMatch;
    uint64_t m_nEmpty;
    uint64_t m_nTooOld;
    uint64_t m_nTimedOut;
    int64_t m_timeDiff;
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
    DrpBase m_drp;
    PvParameters& m_para;
    std::unique_ptr<PvDetector> m_pvDetector;
    Detector* m_det;
    bool m_unconfigure;
};

}

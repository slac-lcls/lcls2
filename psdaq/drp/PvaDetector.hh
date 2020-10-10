#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <mutex>
#include <condition_variable>
#include "DrpBase.hh"
#include "XpmDetector.hh"
#include "spscqueue.hh"
#include "psdaq/epicstools/PvMonitorBase.hh"
#include "psdaq/service/Collection.hh"

namespace Drp {

class PvaDetector;

class PvaMonitor : public Pds_Epics::PvMonitorBase
{
public:
    PvaMonitor(Parameters& para, const std::string& channelName, PvaDetector& det, const std::string& provider) :
      Pds_Epics::PvMonitorBase(channelName, provider),
      m_para(para),
      m_pvaDetector(det)
    {
    }
public:
    void onConnect()    override;
    void onDisconnect() override;
    void updated()      override;
public:
    void getVarDef(XtcData::VarDef&, size_t& payloadSize, size_t rankHack); // Revisit: Hack!
private:
    Parameters&  m_para;
    PvaDetector& m_pvaDetector;
};


class PvaDetector : public XpmDetector
{
public:
    PvaDetector(Parameters& para, const std::string& pvName, DrpBase& drp);
    ~PvaDetector();
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
    void shutdown() override;
    void connect();
    void process(const XtcData::TimeStamp&);
private:
    void _worker();
    void _timeout(const XtcData::TimeStamp& timestamp);
    void _matchUp();
    void _handleMatch(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg);
    void _handleYounger(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg);
    void _handleOlder(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg);
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    enum {RawNamesIndex = NamesIndex::BASE, InfoNamesIndex};
    const std::string m_pvDescriptor;
    DrpBase& m_drp;
    std::shared_ptr<PvaMonitor> m_pvaMonitor;
    std::thread m_workerThread;
    SPSCQueue<uint32_t> m_pgpQueue;
    SPSCQueue<XtcData::Dgram*> m_pvQueue;
    SPSCQueue<XtcData::Dgram*> m_bufferFreelist;
    std::vector<uint8_t> m_buffer;
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
    uint32_t m_firstDimKw;              // Revisit: Hack!
};


class PvaApp : public CollectionApp
{
public:
    PvaApp(Parameters& para, const std::string& pvaName);
    ~PvaApp();
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
    void _shutdown();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);
private:
    DrpBase m_drp;
    Parameters& m_para;
    std::unique_ptr<Detector> m_det;
    bool m_unconfigure;
};

}

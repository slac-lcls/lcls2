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
#include "psdaq/service/Collection.hh"
#include "psdaq/epicstools/PVBase.hh"

namespace Drp {

class PvaDetector;

class PvaMonitor : public Pds_Epics::PVBase
{
public:
    PvaMonitor(const char* channelName, PvaDetector& det) : Pds_Epics::PVBase(channelName), m_pvaDetector(det) {}
    void printStructure();
    XtcData::VarDef get(size_t& payloadSize);
    void updated() override;
public:
    std::function<size_t(void* data, size_t& length)> getData;
private:
    template<typename T> size_t _getDatumT(void* data, size_t& length) {
        *static_cast<T*>(data) = getScalarAs<T>();
        length = 1;
        return sizeof(T);
    }
    template<typename T> size_t _getDataT(void* data, size_t& length) {
        //pvd::shared_vector<const T> vec((T*)data, [](void*){}, 0, 128); // Doesn't work
        pvd::shared_vector<const T> vec;
        getVectorAs<T>(vec);
        length = vec.size();
        size_t size = length * sizeof(T);
        memcpy(data, vec.data(), size);
        return size;
    }
private:
    PvaDetector& m_pvaDetector;
};


class PvaDetector : public XpmDetector
{
public:
    PvaDetector(Parameters& para, const std::string& pvName, DrpBase& drp);
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
    void shutdown() override;
    void reset();
    void process(const PvaMonitor&);
private:
    void _worker();
    void _timeout(const XtcData::TimeStamp& timestamp);
    void _defer(const XtcData::TimeStamp& timestamp);
    bool _handle(const XtcData::TimeStamp& timestamp, unsigned index, const XtcData::Dgram* deferred);
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    enum {PvaNamesIndex = NamesIndex::BASE};
    const std::string& m_pvName;
    DrpBase& m_drp;
    std::unique_ptr<PvaMonitor> m_pvaMonitor;
    std::thread m_workerThread;
    SPSCQueue<uint32_t> m_inputQueue;
    SPSCQueue<XtcData::Dgram*> m_deferredQueue;
    SPSCQueue<XtcData::Dgram*> m_deferredFreelist;
    std::vector<uint8_t> m_deferredBuffer;
    mutable std::mutex m_lock;
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
};


class PvaApp : public CollectionApp
{
public:
    PvaApp(Parameters& para, const std::string& pvaName);
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _shutdown();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);
private:
    DrpBase m_drp;
    Parameters& m_para;
    std::unique_ptr<Detector> m_det;
    bool m_unconfigure;
};

}

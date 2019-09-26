#pragma once

#pragma once

#include <thread>
#include <atomic>
#include "DrpBase.hh"
#include "PGPDetector.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/epicstools/PVBase.hh"

namespace Drp {

class BldDescriptor : public Pds_Epics::PVBase
{
public:
    BldDescriptor(const char* channelName) : Pds_Epics::PVBase(channelName) {}
    XtcData::VarDef get(unsigned& payloadSize);
};

class Bld
{
public:
    Bld(unsigned mcaddr, unsigned port);
    ~Bld();
    uint64_t next(unsigned payloadSize, uint8_t** payload);
    static const unsigned MTU = 8192;
    static const unsigned HeaderSize = 20;
private:
    uint64_t headerPulseId() const {return *reinterpret_cast<const uint64_t*>(m_buffer.data());}
    int m_sockfd;
    unsigned m_bufferSize;
    unsigned m_position;
    bool m_first;
    std::vector<uint8_t> m_buffer;
};

class BldApp : public CollectionApp
{
public:
    BldApp(Parameters& para);
    void shutdown();
    nlohmann::json connectionInfo() override;
    void handleReset(const nlohmann::json& msg) override;
private:
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void connectPgp(const nlohmann::json& json, const std::string& collectionId);
    void worker(std::shared_ptr<MetricExporter> exporter);
    void sentToTeb(XtcData::Dgram& dgram, uint32_t index);

    DrpBase m_drp;
    Parameters& m_para;
    std::thread m_workerThread;
    std::unique_ptr<Pds_Epics::PVBase> m_pvaAddr;
    std::unique_ptr<Pds_Epics::PVBase> m_pvaPort;
    std::unique_ptr<BldDescriptor> m_pvaDescriptor;
    XtcData::NameIndex m_nameIndex;
    std::atomic<bool> m_terminate;
};

}

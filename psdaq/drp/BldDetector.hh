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
    ~BldDescriptor();
    XtcData::VarDef get(unsigned& payloadSize);
};

class Bld
{
public:
    Bld(unsigned mcaddr, unsigned port, unsigned interface, 
        unsigned pulseIdPos, unsigned headerSize, unsigned payloadSize);
    Bld(const Bld&);
    ~Bld();
public:
    static const unsigned MTU = 9000;
    static const unsigned PulseIdPos      =  0; // LCLS-II style
    static const unsigned HeaderSize      = 20;
    static const unsigned DgramPulseIdPos =  8; // LCLS-I style
    static const unsigned DgramHeaderSize = 52;
public:
    uint64_t next       ();
    uint8_t* payload    () const { return m_payload; }
    unsigned payloadSize() const { return m_payloadSize; }
private:
    uint64_t headerPulseId() const {return *reinterpret_cast<const uint64_t*>(m_buffer.data()+m_pulseIdPos);}
    unsigned m_pulseIdPos;
    unsigned m_headerSize;
    unsigned m_payloadSize;
    int      m_sockfd;
    unsigned m_bufferSize;
    unsigned m_position;
    std::vector<uint8_t> m_buffer;
    uint8_t* m_payload;
};

class BldPVA 
{
public:
    BldPVA(const char* name,
           const char* pvname,
           unsigned    interface);
    ~BldPVA();
public:
    std::string                        _name;
    unsigned                           _interface;
    std::shared_ptr<Pds_Epics::PVBase> _pvaAddr;
    std::shared_ptr<Pds_Epics::PVBase> _pvaPort;
    std::shared_ptr<BldDescriptor>     _pvaPayload;
};

class BldFactory
{
public:
    BldFactory(const BldPVA& pva);
    BldFactory(const char* name, unsigned interface);
    BldFactory(const char* name, unsigned interface, 
               unsigned addr, unsigned port, std::shared_ptr<BldDescriptor>);
    BldFactory(const BldFactory&);
    ~BldFactory();
public:
    Bld&               handler   ();
    XtcData::NameIndex addToXtc  (XtcData::Xtc&, 
                                  const XtcData::NamesId&);
private:
    std::string                    _name;
    XtcData::Alg                   _alg;
    XtcData::VarDef                _varDef;
    std::shared_ptr<BldDescriptor> _pvaPayload;
    std::shared_ptr<Bld          > _handler;
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

    DrpBase                                    m_drp;
    Parameters&                                m_para;
    RunInfo                                    m_runInfo;
    XtcData::NamesLookup                       m_namesLookup;
    std::thread                                m_workerThread;
    std::vector<std::shared_ptr<BldFactory> >  m_config;
    std::atomic<bool>                          m_terminate;
    std::shared_ptr<MetricExporter>            m_exporter;
    bool                                       m_unconfigure;
};

}

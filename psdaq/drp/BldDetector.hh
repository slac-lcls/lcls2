#pragma once

#include <thread>
#include <atomic>
#include <string>
#include "DrpBase.hh"
#include "XpmDetector.hh"
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
    BldPVA(std::string det,
           unsigned    interface);
    ~BldPVA();
public:
    std::string                        _detName;
    std::string                        _detType;
    std::string                        _detId;
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
    std::string                    _detName;
    std::string                    _detType;
    std::string                    _detId;
    XtcData::Alg                   _alg;
    XtcData::VarDef                _varDef;
    std::shared_ptr<BldDescriptor> _pvaPayload;
    std::shared_ptr<Bld          > _handler;
};


class Pgp
{
public:
    Pgp(Parameters& para, DrpBase& drp, Detector* det);

    Pds::EbDgram* next(uint64_t pulseId, uint32_t& evtIndex, uint64_t& bytes);
    void worker(std::shared_ptr<Pds::MetricExporter> exporter);
    void shutdown();
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex, uint64_t& bytes);
    void _sendToTeb(Pds::EbDgram& dgram, uint32_t index);
private:
    enum {BldNamesIndex = NamesIndex::BASE}; // Revisit: This belongs in BldDetector
    Parameters&                                m_para;
    DrpBase&                                   m_drp;
    Detector*                                  m_det;
    static const int MAX_RET_CNT_C = 100;
    int32_t                                    dmaRet[MAX_RET_CNT_C];
    uint32_t                                   dmaIndex[MAX_RET_CNT_C];
    uint32_t                                   dest[MAX_RET_CNT_C];
    std::vector<std::shared_ptr<BldFactory> >  m_config;
    std::atomic<bool>                          m_terminate;
    bool                                       m_running;
    int32_t                                    m_available;
    int32_t                                    m_current;
    uint32_t                                   m_lastComplete;
    XtcData::TransitionId::Value               m_lastTid;
    uint32_t                                   m_lastData[6];
    unsigned                                   m_nodeId;
    uint64_t                                   m_next;
};


class BldApp : public CollectionApp
{
public:
    BldApp(Parameters& para);
    ~BldApp() override;
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

    DrpBase                              m_drp;
    Parameters&                          m_para;
    std::thread                          m_workerThread;
    std::unique_ptr<Pgp>                 m_pgp;
    Detector*                            m_det;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    bool                                 m_unconfigure;
};

}

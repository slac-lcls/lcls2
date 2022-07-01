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
        unsigned timestampPos, unsigned headerSize, unsigned payloadSize,
        uint64_t timestampCorr=0);
    Bld(const Bld&);
    ~Bld();
public:
    static const unsigned MTU = 9000;
    static const unsigned PulseIdPos        =  0; // LCLS-II style
    static const unsigned TimestampPos      =  8; // LCLS-II style
    static const unsigned HeaderSize        = 20;
    static const unsigned DgramTimestampPos =  0; // LCLS-I style
    static const unsigned DgramPulseIdPos   =  8; // LCLS-I style
    static const unsigned DgramHeaderSize   = 60;
public:
    bool     ready      () const { return (m_position + m_payloadSize + 4) <= m_bufferSize; }
    void     clear      (uint64_t ts);
    uint64_t next       ();
    uint8_t* payload    () const { return m_payload; }
    unsigned payloadSize() const { return m_payloadSize; }
    unsigned fd         () const { return m_sockfd; }
private:
    uint64_t headerTimestamp  () const {return *reinterpret_cast<const uint64_t*>(m_buffer.data()+m_timestampPos) - m_timestampCorr;}
    uint64_t headerPulseId    () const {return *reinterpret_cast<const uint64_t*>(m_buffer.data()+PulseIdPos);}
    int      m_timestampPos;
    int      m_headerSize;
    int      m_payloadSize;
    int      m_sockfd;
    int      m_bufferSize;
    int      m_position;
    std::vector<uint8_t> m_buffer;
    uint8_t* m_payload;
    uint64_t m_timestampCorr;
    uint64_t m_pulseId;
    unsigned m_pulseIdJump;
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
                                  const void* bufEnd,
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

    Pds::EbDgram* next(uint32_t& evtIndex, uint64_t& bytes); // Slow case
    //  Returns NULL if earliest received data is already later than requested data
    Pds::EbDgram* next(uint64_t timestamp, uint32_t& evtIndex, uint64_t& bytes); // Non-Slow case
    void worker(std::shared_ptr<Pds::MetricExporter> exporter);
    void shutdown();
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex, uint64_t& bytes);
    void _sendToTeb(Pds::EbDgram& dgram, uint32_t index);
    bool _ready() const { return m_current < m_available; }
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
    int64_t                                    m_latency;
    uint64_t                                   m_nDmaRet;
};


class BldApp : public CollectionApp
{
public:
    BldApp(Parameters& para);
    ~BldApp() override;
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo() override;
    void connectionShutdown() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
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

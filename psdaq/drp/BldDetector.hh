#pragma once

#include <thread>
#include <atomic>
#include <string>
#include "DrpBase.hh"
#include "XpmDetector.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include "psdaq/epicstools/PVBase.hh"

#include <chrono>

namespace Drp {

class BldDescriptor : public Pds_Epics::PVBase
{
public:
    BldDescriptor(const char* channelName) : Pds_Epics::PVBase("pva",channelName) {}
    ~BldDescriptor();
    XtcData::VarDef get(unsigned& payloadSize, std::vector<unsigned>& sizes);
};

class Bld
{
public:
    Bld(unsigned mcaddr, unsigned port, unsigned interface,
        unsigned timestampPos, unsigned pulseIdPos,
        unsigned headerSize, unsigned payloadSize,
        uint64_t timestampCorr=0, bool varLenArr=false,
        std::vector<unsigned> entryByteSizes={},      // For varLenArr Bld
        std::map<unsigned,unsigned> arraySizeMap={}); // For varLenArr Bld
    Bld(const Bld&);
    ~Bld();
public:
    static inline constexpr unsigned MTU = 9000;
    static inline constexpr unsigned TimestampPos      =  0; // LCLS-II style
    static inline constexpr unsigned PulseIdPos        =  8; // LCLS-II style
    static inline constexpr unsigned HeaderSize        = 20;
    static inline constexpr unsigned DgramTimestampPos =  0; // LCLS-I style
    static inline constexpr unsigned DgramPulseIdPos   =  8; // LCLS-I style
    static inline constexpr unsigned DgramHeaderSize   = 60;
public:
    bool     ready      () const { return (m_position + m_payloadSize + 4) <= m_bufferSize; }
    void     clear      (uint64_t ts);
    uint64_t next       ();
    uint8_t* payload    () const { return m_payload; }
    unsigned payloadSize() const { return m_payloadSize; }
    unsigned fd         () const { return m_sockfd; }
private:
    uint64_t headerTimestamp  () const {return *reinterpret_cast<const uint64_t*>(m_buffer.data()+m_timestampPos) - m_timestampCorr;}
    uint64_t headerPulseId    () const {return *reinterpret_cast<const uint64_t*>(m_buffer.data()+m_pulseIdPos);}
    void _calcVarPayloadSize  (); // Modifies m_payloadSize
private:
    int      m_timestampPos;
    int      m_pulseIdPos;
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
    bool     m_varLenArr;
    std::vector<unsigned> m_entryByteSizes;
    std::map<unsigned,unsigned> m_arraySizeMap;
};

class BldPVA
{
public:
    BldPVA(std::string det,
           unsigned    interface);
    ~BldPVA();
public:
    std::string     detName() const { return _detName; }
    std::string     detType() const { return _detType; }
    std::string     detId  () const { return _detId; }
    XtcData::Alg    alg    () const { return _alg; }
    unsigned        interface() const { return _interface; }
    bool            ready() const;
    unsigned        addr() const;
    unsigned        port() const;
    XtcData::VarDef varDef(unsigned& sz, std::vector<unsigned>&) const;
private:
    std::string                        _detName;
    std::string                        _detType;
    std::string                        _detId;
    XtcData::Alg                       _alg;
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
    void               addEventData(XtcData::Xtc&,
                                    const void* bufEnd,
                                    XtcData::NamesLookup&,
                                    XtcData::NamesId&);
private:
    std::string                    _detName;
    std::string                    _detType;
    std::string                    _detId;
    XtcData::Alg                   _alg;
    XtcData::VarDef                _varDef;
    std::shared_ptr<BldDescriptor> _pvaPayload;
    std::shared_ptr<Bld          > _handler;
    std::vector<unsigned>          _arraySizes;
    std::map<unsigned,unsigned>    _arraySizeMap;
    std::vector<unsigned>          _entryByteSizes;
    bool                           _varLenArr;
};


class Pgp : public PgpReader
{
public:
    Pgp(Parameters& para, DrpBase& drp, Detector* det);

    const Pds::TimingHeader* next();
    void worker(const std::shared_ptr<Pds::MetricExporter> exporter);
    void shutdown();
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    int  _setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter);
    void _sendToTeb(Pds::EbDgram& dgram, uint32_t index);
    bool _ready() const { return m_current < m_available; }
private:
    enum {BldNamesIndex = NamesIndex::BASE}; // Revisit: This belongs in BldDetector
    Parameters&                                m_para;
    DrpBase&                                   m_drp;
    Detector*                                  m_det;
    static const int MAX_RET_CNT_C = 100;
    std::vector<std::shared_ptr<BldFactory> >  m_config;
    std::atomic<bool>                          m_terminate;
    bool                                       m_running;
    int32_t                                    m_available;
    int32_t                                    m_current;
    uint64_t                                   m_nevents;
    uint64_t                                   m_nmissed;
    uint64_t                                   m_nDmaRet;
    enum TmoState { None, Started, Finished };
    TmoState                                   m_tmoState;
    std::chrono::time_point<Pds::fast_monotonic_clock> m_tInitial;
};


class BldDrp : public DrpBase
{
public:
    BldDrp(Parameters&, MemPoolCpu&, Detector&, ZmqContext&);
    virtual ~BldDrp() {}
    std::string configure(const nlohmann::json& msg);
    unsigned unconfigure();
private:
    Pgp                                  m_pgp;
    std::thread                          m_workerThread;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
};


class BldApp : public CollectionApp
{
public:
    BldApp(Parameters& para);
    ~BldApp() override;
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

    Parameters&               m_para;
    MemPoolCpu                m_pool;
    std::unique_ptr<Detector> m_det;
    std::unique_ptr<BldDrp>   m_drp;
    bool                      m_unconfigure;
};

}

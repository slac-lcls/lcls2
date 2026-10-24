/**
 * @file
 * @brief BLD DRP: receivers for beam-line data multicast streams (Bld), their configuration from built-in tables or EPICS PVs (BldPVA, BldFactory), and the BLD Pgp reader, BldDrp and BldApp.
 */
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

/** Maximum number of BLD sources in one process (16); sizes the per-source counters and metric exporters. */
static const unsigned MAX_BLD = 16;

namespace Drp {

/** PVA channel whose structure describes the payload of a BLD source; used by BldPVA. */
class BldDescriptor : public Pds_Epics::PVBase
{
public:
    /** Connect to channelName with the pva provider. */
    BldDescriptor(const char* channelName) : Pds_Epics::PVBase("pva",channelName) {}
    /** Log a warning. */
    ~BldDescriptor();
    /** Build a VarDef from the PV structure: a UINT64 severity field followed by one entry per field, with payloadSize set to the total byte size and 0 appended to sizes for each scalar. Throws a std::string for a missing structure, a non-fixed array or an unsupported type; the array case has no break, so it also throws for every array field. */
    XtcData::VarDef get(unsigned& payloadSize, std::vector<unsigned>& sizes);
};

/** Receives a BLD multicast stream on a UDP socket and steps through the events packed in each packet (a header, then entries carrying timestamp and pulse ID offsets); can also simulate packets. */
class Bld {
public:
    /** Open a UDP socket with a 16 MiB receive buffer, bind it to mcaddr:port and join that multicast group on interface; throws a std::string on failure. The other arguments give the header layout, the fixed payload size, simulation, a timestamp correction and the variable-length-array description. */
    Bld(unsigned mcaddr, unsigned port, unsigned interface,
        unsigned timestampPos, unsigned pulseIdPos,
        unsigned headerSize, unsigned payloadSize,
        bool simulate,
        uint64_t timestampCorr=0, bool varLenArr=false,
        std::vector<unsigned> entryByteSizes={},      // For varLenArr Bld
        std::map<unsigned,unsigned> arraySizeMap={}); // For varLenArr Bld
    /** Copy only the layout fields and the socket descriptor (other members are left uninitialized) and log an error. */
    Bld(const Bld&);
    /** Close the socket. */
    ~Bld();
public:
    static inline constexpr unsigned MTU = 9000;  ///< Receive buffer size in bytes (9000).
    /** Offset of the timestamp in an LCLS-II style header (0). */
    static inline constexpr unsigned TimestampPos      =  0; // LCLS-II style
    /** Offset of the pulse ID in an LCLS-II style header (8). */
    static inline constexpr unsigned PulseIdPos        =  8; // LCLS-II style
    static inline constexpr unsigned HeaderSize        = 20;  ///< Header size of the LCLS-II style (20 bytes); used for PV-described sources.
    /** Offset of the timestamp in an LCLS-I style header (0). */
    static inline constexpr unsigned DgramTimestampPos =  0; // LCLS-I style
    /** Offset of the pulse ID in an LCLS-I style header (8). */
    static inline constexpr unsigned DgramPulseIdPos   =  8; // LCLS-I style
    static inline constexpr unsigned DgramHeaderSize   = 60;  ///< Header size of the LCLS-I style (60 bytes); used for the built-in sources.
public:
    /** Return true if the current packet still holds another event (a 4-byte offset word plus payload) after the current position. */
    bool     ready      () const { return (m_position + m_payloadSize + 4) <= m_bufferSize; }
    /** Skip events, receiving (or simulating) packets as needed, until one with a timestamp at or after ts is current or no more data is available; dropped timestamps and pulse ID jumps are logged. */
    void     clear      (uint64_t ts);
    /** Advance to the next event and return its timestamp (header timestamp minus the correction, plus the entry offset for packed events), receiving a new packet when the current one is used up. Returns 0 when a new packet is needed but none is available, and always in that case when simulating. */
    uint64_t next       ();
    /** Return the payload of the current event. */
    uint8_t* payload    () const { return m_payload; }
    /** Return the payload size of the current event (recomputed per event for variable-length arrays). */
    unsigned payloadSize() const { return m_payloadSize; }
    //    unsigned fd         () const { return m_sockfd; }
private:
    uint64_t headerTimestamp  () const {return *reinterpret_cast<const uint64_t*>(m_buffer.data()+m_timestampPos) - m_timestampCorr;}
    uint64_t headerPulseId    () const {return *reinterpret_cast<const uint64_t*>(m_buffer.data()+m_pulseIdPos);}
    void _calcVarPayloadSize  (); // Modifies m_payloadSize
private:
    int      m_timestampPos;
    int      m_pulseIdPos;
    int      m_headerSize;
    int      m_payloadSize;
    bool     m_simulate;
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

/** Looks up the multicast address, port and payload description of a BLD source through EPICS PVs: ID:BLD1_MULT_ADDR and ID:BLD1_MULT_PORT (ca) and ID:BLD_PAYLOAD (pva). */
class BldPVA
{
public:
    /** Split det at the plus signs into detector name, type and ID, and create the PV monitors for address, port and payload with the ID as prefix. */
    BldPVA(std::string det,
           unsigned    interface);
    /** Does nothing; empty body. */
    ~BldPVA();
public:
    /** Return the detector name parsed from det. */
    std::string     detName() const { return _detName; }
    /** Return the detector type parsed from det. */
    std::string     detType() const { return _detType; }
    /** Return the detector ID parsed from det (also the PV prefix). */
    std::string     detId  () const { return _detId; }
    /** Return the algorithm (raw, version 1.0.0). */
    XtcData::Alg    alg    () const { return _alg; }
    /** Return the interface address given to the constructor. */
    unsigned        interface() const { return _interface; }
    /** Log the state of the three PVs and return true if all of them are ready. */
    bool            ready() const;
    /** Return the address PV parsed as a dotted IPv4 address, in host byte order (0 if it does not parse). */
    unsigned        addr() const;
    /** Return the port PV value. */
    unsigned        port() const;
    /** Return BldDescriptor::get() for the payload PV, filling sz and the array sizes. */
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

/** Creates the Bld receiver and the XTC names of one BLD source, either from a built-in table or from EPICS PVs (BldPVA). */
class BldFactory
{
public:
    /** Wait (polling every 10 ms) until the PVs of pva are ready, then create a Bld for its multicast address and port with the LCLS-II header layout and the payload described by its PV. */
    BldFactory(const BldPVA& pva, bool simulate);
    /** Choose the multicast address, names and algorithm for a known BLD name (text after the last colon): ebeam, pcav, gasdet, gmd, xgmd, feespec, or a beam monitor or UsdUsb name from BldNames; then create a Bld on port 10148 with the LCLS-I header layout. Throws a std::string for an unknown name. */
    BldFactory(const char* name, unsigned interface, bool simulate);
    // BldFactory(const char* name, unsigned interface,
    //            unsigned addr, unsigned port, std::shared_ptr<BldDescriptor>,
    //            bool simulate);
    /** Copy only the name, type, ID and algorithm (not the receiver or names) and log a warning. */
    BldFactory(const BldFactory&);
    /** Does nothing; empty body. */
    ~BldFactory();
public:
    /** Return the Bld receiver. */
    Bld&               handler   ();
    /** Add Names for this source with namesId to xtc and return their NameIndex. */
    XtcData::NameIndex addToXtc  (XtcData::Xtc&,
                                  const void* bufEnd,
                                  const XtcData::NamesId&);
    /** Copy the current Bld payload into a DescribedData entry and set the array shapes (fixed sizes, or sizes read from the payload for variable-length arrays). */
    void               addEventData(XtcData::Xtc&,
                                    const void* bufEnd,
                                    XtcData::NamesLookup&,
                                    XtcData::NamesId&);
    /** Return the field definitions of this source. */
    XtcData::VarDef&   varDef() { return _varDef; }
    /** Return the detector name. */
    const std::string& detName() const { return _detName; }
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
    /** Construct the PgpReader on the pool of drp (at most 100 buffers per read, 32 per free batch) and set the driver mask for laneMask on the virtual channel of det, logging an error on failure. */
    Pgp(Parameters& para, DrpBase& drp, Detector* det);

    /** Return the TimingHeader of the current DMA buffer, reading new buffers when the previous batch is used up and timing out TEB batches while waiting. Returns nullptr if nothing arrives within 20 ms. */
    const Pds::TimingHeader* next();
    /** Thread body: build the BLD sources from Parameters::detType (comma-separated names, or entries with plus signs looked up through PVs), set up their metrics, then match timing headers with BLD events by timestamp. Transitions are sent at once (Configure also adds the source names); L1Accepts are sent once older than bld_tmo_ms (default 50 ms), with DroppedContribution damage if a source has no matching event. */
    void worker(std::shared_ptr<Pds::MetricExporter> exporter[], prometheus::Exposer*);
    /** Ask the worker loop to stop and clear the names lookup of the detector. */
    void shutdown();
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    int  _setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter);
    void _sendToTeb(Pds::EbDgram& dgram, uint32_t index);
    bool _ready() const { return m_current < m_available; }
private:
    Parameters&                                m_para;
    DrpBase&                                   m_drp;
    Detector*                                  m_det;
    static const unsigned MAX_RET_CNT_C = 100;
    std::vector<std::shared_ptr<BldFactory> >  m_config;
    std::atomic<bool>                          m_terminate;
    bool                                       m_running;
    int32_t                                    m_available;
    int32_t                                    m_current;
    uint64_t                                   m_nevents;
    uint64_t                                   m_nmissed;
    uint64_t                                   m_nDmaRet;
    uint64_t                                   m_damage[MAX_BLD];
    uint64_t                                   m_events[MAX_BLD];
    enum TmoState { None, Started, Finished };
    TmoState                                   m_tmoState;
    std::chrono::time_point<Pds::fast_monotonic_clock> m_tInitial;
};


/** DrpBase of the BLD process; runs the BLD Pgp worker thread. */
class BldDrp : public DrpBase
{
public:
    /** Construct DrpBase and the BLD Pgp reader, and install a TebReceiver, or a CubeTebReceiver when cube workers are configured. */
    BldDrp(Parameters&, MemPoolCpu&, Detector&, ZmqContext&);
    /** Does nothing; empty body. */
    virtual ~BldDrp() {}
    /** Call DrpBase::configure() and return its result (empty on success). */
    std::string configure(const nlohmann::json& msg);
    /** Call DrpBase::unconfigure(), stop and join the worker thread and release the metric exporters; returns 0. */
    unsigned unconfigure();
    /** Call DrpBase::startup() and start the worker thread; returns an empty string or the DrpBase error. */
    std::string startup(XtcData::Xtc& xtc, const void* be);
private:
    Pgp                                  m_pgp;
    std::thread                          m_workerThread;
    std::shared_ptr<Pds::MetricExporter> m_exporter[MAX_BLD];
};


/** CollectionApp of the BLD DRP process: owns the MemPoolCpu, a BLD detector and the BldDrp. */
class BldApp : public CollectionApp
{
public:
    /** Register with the collection as a drp, open the PGP pool, initialize Python, and create the BLD detector and the BldDrp. */
    BldApp(Parameters& para);
    /** Call handleReset() and finalize Python. */
    ~BldApp() override;
    /** Unsubscribe from the partition, unconfigure, disconnect and shut down the connection. */
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connectionShutdown() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    std::string _endrun(const nlohmann::json& phase1Info);
    void _unconfigure();
    void _disconnect();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);

    Parameters&               m_para;
    MemPoolCpu                m_pool;
    std::unique_ptr<Detector> m_det;
    std::unique_ptr<BldDrp>   m_drp;
    bool                      m_unconfigure;
    std::string               m_lastKey;
};

}

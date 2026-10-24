/**
 * @file
 * @brief PV DRP: a PGP reader, PvMonitor (matches EPICS PV updates to L1Accepts by timestamp), PvDetector, PvDrp and PvApp.
 */
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
#include "psdaq/service/fast_monotonic_clock.hh"

namespace Drp {

struct PvParameters;
class  PvDetector;

/** PgpReader for the PV DRP that turns complete PGP events into EbDgrams in the pebble. */
class Pgp : public PgpReader
{
public:
    /** Construct the PgpReader (at most 100 buffers per read, 32 per free batch) and set the driver mask for laneMask on the virtual channel of det, logging an error on failure. */
    Pgp(const Parameters& para, MemPool& pool, Detector* det);
    /** Return the next complete event as an EbDgram built in its pebble buffer (its index is stored in evtIndex), reading more DMA buffers when the previous batch is used up. Returns nullptr if the read returned 0 or the buffer did not complete an event. */
    Pds::EbDgram* next(uint32_t& evtIndex);
    /** Return the number of DMA buffers returned by the most recent read. */
    const uint64_t nDmaRet() const { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    Detector* m_det;
    static const unsigned MAX_RET_CNT_C = 100;
    int32_t m_available;
    int32_t m_current;
    uint64_t m_nDmaRet;
};


/** Monitors one EPICS PV and copies its value into the L1Accept whose timestamp equals the timestamp of the PV update. L1Accepts wait in l1Queue; matched (or MissingData-damaged) ones are passed on through pvQueue. */
class PvMonitor : public Pds_Epics::PvMonitorBase
{
public:
    /** Set up the PV monitor (pvName via provider with request and field) and the queues of nBuffers entries, keep the expected type, element count, rank and first-dimension override (-1 values mean take them from the PV), and connect the error PUSH socket to the collection server. */
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
    /** On the first connection, take the type, element count and rank from the PV for any that were not given and mark the monitor ready; aborts if given and PV values differ. With verbose above 1, print the PV structure. */
    void onConnect()    override;
    /** Mark the monitor not ready and send a warning that the PV disconnected. */
    void onDisconnect() override;
    /** When ready and running, compare the update timestamp with queued L1Accepts: on equal timestamps copy the value into that event and pass it on; older L1Accepts are damaged with MissingData and passed on; a younger L1Accept stays queued for the next update. */
    void updated()      override;
public:
    /** Remove dgram from l1Queue if it is the oldest entry. */
    void timeout(Pds::EbDgram*);
    /** Start both queues. */
    void startup();
    /** Shut down both queues and reset the counters. */
    void shutdown();
    /** Wait up to 3 s for the PV to connect, then return its field name, XTC type and rank (rank 2 when a first-dimension override is set). Returns 1 if the parameters are unknown or the payload would not fit in a buffer (a message is sent), else 0. */
    int  getParams(std::string& fieldName, XtcData::Name::DataType& xtcType, int& rank);
    /** Return the index of this PV within the detector. */
    unsigned id() const { return m_id; }
    /** Return the alias under which the PV is recorded. */
    const std::string& alias() const { return m_alias; }
    /** Return the number of PV updates received. */
    uint64_t nUpdates() const { return m_nUpdates; }
    /** Return the number of L1Accepts that got PV data. */
    uint64_t nMatch()   const { return m_nMatch; }
    /** Return the number of L1Accepts passed on without PV data (older than the update). */
    uint64_t nEmpty()   const { return m_nEmpty; }
    /** Return the number of updates older than the oldest queued L1Accept. */
    uint64_t nTooOld()  const { return m_nTooOld; }
    /** Return the last L1Accept minus PV timestamp difference in nanoseconds. */
    int64_t  timeDiff() const { return m_timeDiff; }
    /** Return the latency of the last PV update in microseconds. */
    int64_t  latency()  const { return m_latency; }
private:
    void _event(Pds::EbDgram*, const void* bufEnd);
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
    SPSCQueue<Pds::EbDgram*>        pvQueue;  ///< Events with PV data added (or damaged with MissingData), consumed by the PvDrp collector.
    SPSCQueue<Pds::EbDgram*>        l1Queue;  ///< L1Accept events waiting for a PV update with a matching timestamp.
private:
    static std::mutex               m_commonMutex;
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


/** XpmDetector that records one or more EPICS PVs (Parameters pvSpecs) through PvMonitor objects, with optional configuration from the Python module psdaq.configdb.pvadetector_config. */
class PvDetector : public XpmDetector
{
public:
    /** Construct the XpmDetector base with virtual channel 0 and import psdaq.configdb.pvadetector_config (aborting if that fails). */
    PvDetector(PvParameters&, MemPoolCpu&);
    /** Release the Python module. */
    ~PvDetector();
    /** Call XpmDetector::connect() and create one PvMonitor per PV spec, parsing an optional alias=, provider/, ,firstDim, .field, [shape] and (type) from each spec. Returns 0, or 1 with msg set (and no monitors kept) if a spec has an unrecognized type. */
    unsigned connect(const nlohmann::json&, const std::string& collectionId, std::string& msg);
    /** Call XpmDetector::shutdown() and drop the PV monitors; returns 0. */
    unsigned disconnect();
    /** Run XpmDetector::configure, then for each PV monitor add its raw names (version from the data_vsn kwarg, default 0x010000) and, unless detType is pv, the XTC translation of the JSON returned by Python pvadetector_config (nothing when it returns None). Then add the pvdetinfo names and data listing the PV aliases (comma-separated) and PV names (newline-separated). Returns 0; 1 if the base configure or a monitor's getParams fails, -1 (as unsigned) if the JSON translation fails, and aborts if the config XTC exceeds maxTrSize. */
    unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
    /** Clear the names lookup table; returns 0. */
    unsigned unconfigure();
    using Detector::enable;             // Avoid 'hidden' warning
    /** Set the running flag that lets PV updates be matched. */
    void enable();
    using Detector::disable;            // Avoid 'hidden' warning
    /** Clear the running flag. */
    void disable();
    /** Does nothing; empty body (marked unused). */
    void event(XtcData::Dgram& evt, const void* bufEnd, PGPEvent*, uint64_t l1count) override { /* unused */ };
    /** Does nothing; empty body (marked unused). */
    void event(XtcData::Dgram& evt, const void* bufEnd, const Pds::Eb::ResultDgram&) override { /* unused */ };
    /** Return the PV monitors created by connect(). */
    const std::vector< std::shared_ptr<PvMonitor> >& pvMonitors() const { return m_pvMonitors; }
public:
    static const unsigned maxSupportedPVs = 64;  ///< Maximum number of PVs per detector (64); spaces the names indices below.
    /** NamesId index bases, one block of maxSupportedPVs indices each. */
    enum {
      ConfigNamesIndex = NamesIndex::BASE,  ///< Configuration names, from NamesIndex::BASE (0) plus the PV index.
      RawNamesIndex = unsigned(ConfigNamesIndex) + maxSupportedPVs,  ///< Raw PV data names, from 64 plus the PV index.
      InfoNamesIndex = unsigned(RawNamesIndex) + maxSupportedPVs,  ///< The pvdetinfo names (128).
    };
private:
    std::atomic<bool>                         m_running;
    std::vector< std::shared_ptr<PvMonitor> > m_pvMonitors;
    PyObject*                                 m_pyModule;
    std::string                               m_connectJson;
};


/** DrpBase of the PV DRP: a reader thread hands L1Accepts to the PV monitors and a collector thread sends the results to the TEB. */
class PvDrp : public DrpBase
{
public:
    /** Construct DrpBase, the PGP reader and the event queue, and install a TebReceiver. */
    PvDrp(PvParameters&, MemPoolCpu&, PvDetector&, ZmqContext&);
    /** Does nothing; empty body. */
    virtual ~PvDrp() {}
    /** Call DrpBase::configure() and return its result (empty on success). */
    std::string configure(const nlohmann::json& msg);
    /** Call DrpBase::unconfigure(), then stop and join the reader and collector threads; returns 0. */
    unsigned unconfigure();
    /** Call DrpBase::startup(), start the event queue and the reader and collector threads; returns an empty string or the DrpBase error. */
    std::string startup(XtcData::Xtc& xtc, const void* be);
private:
    int  _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
    void _reader();
    void _collector();
    void _handleTransition(Pds::EbDgram& evtDg, Pds::EbDgram& trDg);
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    using tp_t = std::chrono::time_point<Pds::fast_monotonic_clock>;
    struct it_t
    {
      unsigned index;  ///< Pebble buffer index of the queued event.
      tp_t     t0;  ///< Time the event was queued; the collector uses it for the match_tmo_ms timeout.
    };
    const PvParameters& m_para;
    PvDetector&         m_det;
    Pgp                 m_pgp;
    std::thread         m_readerThread;
    std::thread         m_collectorThread;
    SPSCQueue<it_t>     m_evtQueue;
    std::atomic<bool>   m_terminate;
    uint64_t            m_nEvents;
    uint64_t            m_nUpdates;
    uint64_t            m_nMatch;
    uint64_t            m_nEmpty;
    uint64_t            m_nTooOld;
    uint64_t            m_nTimedOut;
    int64_t             m_timeDiff;
    int64_t             m_age;
};


/** CollectionApp of the PV DRP process: owns the MemPoolCpu, the PvDetector and the PvDrp. */
class PvApp : public CollectionApp
{
public:
    /** Register with the collection as a drp, open the PGP pool, initialize Python, and create the PvDetector and the PvDrp. */
    PvApp(PvParameters& para);
    /** Call handleReset() and finalize Python. */
    ~PvApp();
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
private:
    PvParameters&               m_para;
    MemPoolCpu                  m_pool;
    std::unique_ptr<PvDetector> m_det;
    std::unique_ptr<PvDrp>      m_drp;
    bool                        m_unconfigure;
    std::string                 m_lastKey;
};

}

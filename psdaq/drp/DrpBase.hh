/**
 * @file
 * @brief DrpBase and its helpers: run, chunk and file parameters, the TEB result receiver base, the PGP DMA reader, and the common transition handling of a DRP process.
 */
#pragma once

#include "drp.hh"
#include "FileWriter.hh"
#include "Detector.hh"
#include "psdaq/trigger/src/TriggerPrimitive.hh"
#include "psdaq/trigger/src/utilities.hh"
#include "psdaq/eb/src/TebContributor.hh"
#include "psdaq/eb/src/MebContributor.hh"
#include "psdaq/eb/src/EbCtrbInBase.hh"
#include "psdaq/eb/src/ResultDgram.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include "psdaq/aes-stream-drivers/DmaDriver.h"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/TransitionId.hh"
#include <nlohmann/json.hpp>

namespace Pds {
    class TimingHeader;
}

namespace Drp {

/** ANSI escape sequence that turns terminal text red; used in a few error messages. */
static const char* const RED_ON  = "\033[0;31m";
/** ANSI escape sequence that resets terminal text attributes. */
static const char* const RED_OFF = "\033[0m";


/** Experiment name and run number of the current run, filled by DrpBase::beginrun(). */
struct RunInfo
{
    std::string experimentName;  ///< Experiment name from run_info.experiment_name of the BeginRun phase-1 info (empty if absent).
    uint32_t runNumber;  ///< Run number from run_info.run_number; 0 means not recording (DrpBase::beginrun() also forces 0 for monitor-only detectors).
};

/** Name and ID of the next data file chunk, filled by DrpBase::enable(). */
struct ChunkInfo
{
    std::string filename;  ///< File name of the new chunk (run name plus .xtc2).
    uint32_t chunkId;  ///< Chunk number of the new chunk.
};

/** Values needed to build data file names for a run: output directory, instrument, run number, experiment name, host name, node ID and chunk ID. */
class FileParameters
{
    std::string m_outputDir;
    std::string m_instrument;
    unsigned m_runNumber;
    std::string m_experimentName;
    std::string m_hostname;
    unsigned m_nodeId;
    unsigned m_chunkId;

public:
    /** Set every member to the placeholder value 777 (strings to the text 777). */
    FileParameters() {
        m_outputDir = "777";
        m_instrument = "777";
        m_runNumber = 777;
        m_experimentName = "777";
        m_hostname = "777";
        m_nodeId = 777;
        m_chunkId = 777;
    }

    /** Take the output directory and instrument from para and the run number and experiment name from runInfo, and store hostname and nodeId; the chunk ID starts at 0. */
    FileParameters(const Parameters& para, const RunInfo& runInfo, std::string hostname, unsigned nodeId) {
        m_outputDir = para.outputDir;
        m_instrument = para.instrument;
        m_runNumber = runInfo.runNumber;
        m_experimentName = runInfo.experimentName;
        m_hostname = hostname;
        m_nodeId = nodeId;
        m_chunkId = 0;
    }

    /** Increment the chunk ID; always returns true. */
    bool advanceChunkId()                { ++m_chunkId; return true; }
    // getters
    /** Return the output directory. */
    const std::string& outputDir()       const { return m_outputDir; }
    /** Return the instrument name. */
    const std::string& instrument()      const { return m_instrument; }
    /** Return the run number. */
    unsigned runNumber()                 const { return m_runNumber; }
    /** Return the experiment name. */
    const std::string& experimentName()  const { return m_experimentName; }
    /** Return the host name given to the constructor. */
    const std::string& hostname()        const { return m_hostname; }
    /** Return the node ID given to the constructor. */
    unsigned nodeId()                    const { return m_nodeId; }
    /** Return the current chunk ID. */
    unsigned chunkId()                   const { return m_chunkId; }
    /** Return the run name experiment-rNNNN-sNNN-cNNN, built from the experiment name, run number (4 digits), node ID (3 digits) and chunk ID (3 digits), zero-padded. */
    std::string runName() const;
};

class PgpReader;
class DrpBase;
/** Return a JSON message whose key is pulseId and whose body holds pulseId. */
nlohmann::json createPulseIdMsg(uint64_t pulseId);

/** Base of the DRP trigger result receivers. process() checks each result against the pebble datagram, transfers the result damage, requests file chunking and calls complete(); the class also opens and closes the data and small-data files. Subclasses provide the writers, metrics and complete(). */
class TebReceiverBase : public Pds::Eb::EbCtrbInBase
{
public:
  /** Construct the EbCtrbInBase from the TEB parameters of the DRP, keep references to the DRP and its pool, and allocate a buffer of para.maxTrSize bytes for the Configure datagram. */
  TebReceiverBase(const Parameters&, DrpBase&);
    /** Does nothing; empty body. */
    virtual ~TebReceiverBase() {}
protected:
    void process(const Pds::Eb::ResultDgram&, unsigned index) override;
public:
    /** Reset the EbCtrbInBase counters, the damage count and histogram and the latency; with all true, also reset the expected next buffer index so that index 0 is expected. */
    void resetCounters(bool all);
    /** Register metrics (if the exporter is non-null), remember the node ID on the timing-system DRP (detType ts), and connect to the TEBs with EbCtrbInBase::connect(). Returns 0 or the first error. */
    int  connect(const std::shared_ptr<Pds::MetricExporter>);
    /** Close the files (in case BeginRun failed), then call EbCtrbInBase::unconfigure(). */
    void unconfigure();
    /** If runInfo.runNumber is non-zero, create the directories and open the data file outputDir/instrument/experiment/xtc/RUN.xtc2 and the small-data file RUN.smd.xtc2 in its smalldata subdirectory (RUN is experiment-rNNNN-sNNN-c000), sending a file report message for each. Returns an empty string on success, after which writing() is true, or a message naming the file that failed. */
    std::string openFiles(const RunInfo& runInfo);
    /** If no chunk change is pending, increment the chunk ID in the file parameters, mark a change pending and return true; otherwise return false. */
    bool advanceChunkId();
    /** Close the current data file and open the data file for the new chunk (name from FileParameters::runName()), sending a file report message and clearing the pending flag on success. Returns an empty string on success or an error message (also when not writing). */
    std::string reopenFiles();
    /** If writing, clear the writing flag and close the small-data and data files. Always returns an empty string. */
    std::string closeFiles();
    /** Return the number of bytes recorded since the current chunk started. */
    uint64_t chunkSize() const { return m_offset - m_chunkOffset; }
    /** Set the chunk-request flag. */
    void chunkRequestSet();
    /** Clear the chunk offset, the chunk-request flag and the pending flag left from a previous run. */
    void chunkReset();
    /** Set the recorded byte offset to 0. */
    void offsetReset() { m_offset = 0; }
    /** Add size to the recorded byte offset. */
    void offsetAppend(size_t size) { m_offset += size; }
    /** Return true while files are open for recording (from a successful openFiles() until closeFiles()). */
    bool writing() const { return m_writing; }
    /** Chunk threshold of 500 * 1024^3 bytes (500 GB per the comment): process() requests a new chunk when chunkSize() exceeds it, and DrpBase::enable() advances the chunk once chunkSize() exceeds half of it. */
    static const uint64_t DefaultChunkThresh = 500ull * 1024ull * 1024ull * 1024ull;    // 500 GB
    /** Return the file parameters cached by openFiles(); valid only after a successful openFiles(). */
    const FileParameters& fileParameters() const { return *m_fileParameters; }
    /** Pure virtual: return the writer used for the data file. */
    virtual FileWriterBase& fileWriter() = 0;
    /** Pure virtual: return the writer used for the small-data file. */
    virtual SmdWriterBase& smdWriter() = 0;
protected:
    virtual int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                             std::map<std::string, std::string>& labels) = 0;
    virtual void complete(unsigned index, const Pds::Eb::ResultDgram& result) = 0;
private:
    int _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
protected:
    MemPool& m_pool;
    DrpBase& m_drp;
    unsigned m_tsId;
private:
    bool m_writing;
protected:
    ZmqSocket& m_inprocSend;
private:
    uint32_t m_lastIndex;
    uint64_t m_lastPid;
    XtcData::TransitionId::Value m_lastTid;
    uint32_t m_lastEnv;
    uint64_t m_offset;
    uint64_t m_chunkOffset;
    bool m_chunkPending;
protected:
    bool m_chunkRequest;
    unsigned m_configureIndex;
    std::vector<uint8_t> m_configureBuffer;
    uint64_t m_evtSize;
    uint64_t m_latPid;
    int64_t m_latency;
private:
    uint64_t m_damage;
    std::shared_ptr<Pds::PromHistogram> m_dmgType;
    std::unique_ptr<FileParameters> m_fileParameters;
    const Parameters& m_para;
};

/** Reads DMA buffers from the PGP driver in bulk and assembles them into PGPEvents (one buffer per lane). It checks event counters, readout groups and buffer sanity, and allocates pebble and transition buffers for complete events. */
class PgpReader
{
public:
    /** Size the bulk-read arrays for maxRetCnt buffers and the free-index batch for dmaFreeCnt buffers (aborting if the pool has fewer DMA buffers than dmaFreeCnt), set up poll() on the device file descriptor and reset the pool counters. */
    PgpReader(const Parameters& para, MemPool& pool, unsigned maxRetCnt, unsigned dmaFreeCnt);
    /** Call flush(). */
    virtual ~PgpReader();
    /** Read up to maxRetCnt DMA buffers. If the poll timeout is non-zero (100 ms initially, later 1 ms), first wait in poll(); after a successful read the timeout becomes 0 (pure polling) when buffers arrive faster than one per millisecond, else 1 ms. After an empty read, sleep with exponential back-off up to 1024 us. Returns the readDma() result. */
    int32_t read();
    /** Return the DMA indices queued for freeing, return every DMA buffer that can still be read without counting it, and free all pebble buffers in use. */
    void flush();
    /** Process DMA buffer current of the last read() for detector det: record it in the PGPEvent chosen by its event counter and, once all lanes have arrived, check the event (counter jumps, common readout group, readout groups of SlowUpdates) and allocate a transition buffer (non-L1Accept) and a pebble buffer index. Returns the TimingHeader of a complete, accepted event, otherwise nullptr; aborts on a DMA overflow or a header with zero pulse ID, timestamp or env. */
    const Pds::TimingHeader* handle(Detector* det, unsigned current);
    /** Queue the DMA buffers of all lanes of event for return to the driver (returned in batches of dmaFreeCnt) and clear the lane mask of the event. Thread-safe. */
    void freeDma(PGPEvent* event);
    /** Hook called for events that are dropped; the default does nothing. */
    virtual void handleBrokenEvent(const PGPEvent& event) {}
    /** Reset the last completed event counter to 0; handle() calls this at BeginRun. */
    virtual void resetEventCounter() { m_lastComplete = 0; } // EvtCounter reset
    /** Return the total number of DMA bytes handled. */
    uint64_t dmaBytes()     const { return m_dmaBytes; }
    /** Return the size of the last DMA buffer handled. */
    uint64_t dmaSize()      const { return m_dmaSize; }
    /** Return the last measured TimingHeader latency in microseconds, updated about every 92857 pulse IDs (10 Hz per the code comment). */
    int64_t  latency()      const { return m_latency; }
    /** Return the number of DMA buffers that reported an error. */
    uint64_t nDmaErrors()   const { return m_nDmaErrors; }
    /** Return the number of events dropped because the common readout group was missing. */
    uint64_t nNoComRoG()    const { return m_nNoComRoG; }
    /** Return the number of SlowUpdates dropped because readout groups in Parameters::rogMask were missing. */
    uint64_t nMissingRoGs() const { return m_nMissingRoGs; }
    /** Return the number of timing headers with the error bit set. */
    uint64_t nTmgHdrError() const { return m_nTmgHdrError; }
    /** Return the number of event counter jumps detected. */
    uint64_t nPgpJumps()    const { return m_nPgpJumps; }
    /** Return the number of events dropped because no transition buffer was available. */
    uint64_t nNoTrDgrams()  const { return m_nNoTrDgrams; }
    /** Return dmaGetRxBuffinUserCount() for the device of the pool. */
    int64_t  nPgpInUser()   const { return dmaGetRxBuffinUserCount  (m_pool.fd()); }
    /** Return dmaGetRxBuffinHwCount() for the device of the pool. */
    int64_t  nPgpInHw()     const { return dmaGetRxBuffinHwCount    (m_pool.fd()); }
    /** Return dmaGetRxBuffinPreHwQCount() for the device of the pool. */
    int64_t  nPgpInPreHw()  const { return dmaGetRxBuffinPreHwQCount(m_pool.fd()); }
    /** Return dmaGetRxBuffinSwQCount() for the device of the pool. */
    int64_t  nPgpInRx()     const { return dmaGetRxBuffinSwQCount   (m_pool.fd()); }
    /** Return the latency of timestamp time (Pds::Eb::latency()) in nanoseconds. */
    std::chrono::nanoseconds age(const XtcData::TimeStamp& time) const;
private:
    void _setTimeOffset(const XtcData::TimeStamp& time);
protected:
    const Parameters& m_para;
    MemPool& m_pool;
    pollfd m_pfd;
    Pds::fast_monotonic_clock::time_point m_t0;
    int m_tmo;
    unsigned m_us;
    std::vector<int32_t> dmaRet;
    std::vector<uint32_t> dmaIndex;
    std::vector<uint32_t> dest;
    std::vector<uint32_t> dmaFlags;
    std::vector<uint32_t> dmaErrors;
    uint32_t m_lastComplete;
    XtcData::TransitionId::Value m_lastTid;
    uint32_t m_lastData[6];
    std::vector<uint32_t> m_dmaIndices;
    unsigned m_count;
    uint64_t m_dmaBytes;
    uint64_t m_dmaSize;
    uint64_t m_latPid;
    int64_t m_latency;
    uint64_t m_nDmaErrors;
    uint64_t m_nNoComRoG;
    uint64_t m_nMissingRoGs;
    uint64_t m_nTmgHdrError;
    uint64_t m_nPgpJumps;
    uint64_t m_nNoTrDgrams;
    std::mutex m_lock;
    bool m_dmaOverrun;
};

class PV;

/** Common machinery of a DRP process: TEB and MEB contributors, the TEB result receiver, the trigger primitive, Prometheus metrics, and the handling of the connect, configure, run and chunking transitions. */
class DrpBase
{
public:
    /** Create the Prometheus exposer (port offset from the lowest laneMask bit and the numeric suffix of the device name), create the TEB and MEB contributors from para, and connect the inproc://drp PAIR socket. Also stat()s the output directory and appends the pva_addr kwarg to EPICS_PVA_ADDR_LIST. */
    DrpBase(Parameters& para, MemPool& pool, Detector& det, ZmqContext& context);
protected:
    void setTebReceiver(std::unique_ptr<TebReceiverBase> tebRecv) { m_tebReceiver = std::move(tebRecv); }
public:
    /** Call disconnect(), then shut down the TEB contributor, the MEB contributor and the TEB receiver. */
    void shutdown();
    /** Start listening for TEB results on ip with an ephemeral port (aborting on failure) and return JSON with drp_port, num_buffers, max_ev_size and max_tr_size. */
    nlohmann::json connectionInfo(const std::string& ip);
    /** Keep msg and id, parse the connection parameters, (re)create the metric exporter, then connect the TEB contributor, the MEB contributor (only if MEBs are listed) and the TEB receiver. Returns an empty string or an error message. */
    std::string connect(const nlohmann::json& msg, size_t id);
    /** Load the trigger primitive, configure the TEB contributor and the MEB contributor (if MEBs are listed), configure the trigger primitive (if this DRP provides trigger input) and the TEB receiver. Returns an empty string or an error message. */
    std::string configure(const nlohmann::json& msg);
    /** Fill runInfo from run_info in phase1Info (run number forced to 0 if monitor_info marks this detector monitor-only), open the files if the run number is non-zero, and reset the contributor and receiver counters and the chunk state. Returns an empty string or an error message. */
    std::string beginrun(const nlohmann::json& phase1Info, RunInfo& runInfo);
    /** Does nothing; returns an empty string. */
    std::string endrun(const nlohmann::json& phase1Info);
    /** If writing and the current chunk exceeds half of DefaultChunkThresh, advance the chunk ID; if that succeeds, set chunkRequest and fill chunkInfo with the new file name and chunk ID. Always returns an empty string. */
    std::string enable(const nlohmann::json& phase1Info, bool& chunkRequest, ChunkInfo& chunkInfo);
    /** Unconfigure the TEB contributor, the MEB contributor (if MEBs are listed) and the TEB receiver. */
    void unconfigure();
    /** Call unconfigure(), disconnect the TEB contributor, the MEB contributor (if MEBs are listed) and the TEB receiver, and drop the metric exporter. */
    void disconnect();
    /** Print the parameters, let the trigger primitive (if any) add to the Configure data, start the TEB receiver thread and reset the counters. Returns an empty string. */
    std::string startup  (XtcData::Xtc& xtc, const void* bufEnd);
    /** Add the runinfo Names (RunInfoDef, NamesIndex::RUNINFO) to xtc and register them in namesLookup. */
    void runInfoSupport  (XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesLookup& namesLookup);
    /** Add the experiment name and run number of runInfo as runinfo data. */
    void runInfoData     (XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesLookup& namesLookup, const RunInfo& runInfo);
    /** Add the chunkinfo Names (ChunkInfoDef, NamesIndex::CHUNKINFO) to xtc and register them in namesLookup. */
    void chunkInfoSupport(XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesLookup& namesLookup);
    /** Add the file name and chunk ID of chunkInfo as chunkinfo data. */
    void chunkInfoData   (XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesLookup& namesLookup, const ChunkInfo& chunkInfo);
    /** Return the detector given to the constructor. */
    Detector& detector() const {return m_det; }
    /** Return the TEB contributor created by the constructor. */
    Pds::Eb::TebContributor& tebContributor() const {return *m_tebContributor;}
    /** Return the MEB contributor created by the constructor. */
    Pds::Eb::MebContributor& mebContributor() const {return *m_mebContributor;}
    /** Return the TEB result receiver set by a subclass with setTebReceiver(). */
    TebReceiverBase& tebReceiver() const {return *m_tebReceiver;}
    /** Return the trigger primitive created by configure(), or nullptr if this DRP provides no trigger input. */
    Pds::Trg::TriggerPrimitive* triggerPrimitive() const {return m_triggerPrimitive;}
    /** Return the Prometheus exposer created by the constructor. */
    prometheus::Exposer* exposer() const { return m_exposer.get(); }
    /** Return the inproc://drp PAIR socket used to send file report and chunk request messages. */
    ZmqSocket& inprocSend() { return m_inprocSend; }
    /** Return the connect message saved by connect(). */
    const nlohmann::json& connectMsg() const { return m_connectMsg; }
    /** Return the collection ID saved by connect(). */
    size_t collectionId() const { return m_collectionId; }
    /** Return the ID of this DRP (drp_id from the connect message). */
    unsigned nodeId() const {return m_nodeId;}
    /** Return the TEB contributor parameters. */
    const Pds::Eb::TebCtrbParams& tebPrms() const {return m_tPrms;}
    /** Return true if this DRP is the supervisor, chosen while parsing the connect message as the first DRP whose alias contains this detector name. */
    bool isSupervisor() const {return m_isSupervisor;}
    /** Return the IP and port of the supervisor (port 32256 + 8 * xpm_id + partition). */
    const std::string& supervisorIpPort() const {return m_supervisorIpPort;}
    MemPool& pool;  ///< Buffer pools shared with the detector code.
private:
    int setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter);
    int setupTriggerPrimitives(const nlohmann::json& body);
    int parseConnectionParams(const nlohmann::json& body, size_t id);
    void printParams() const;
    Parameters& m_para;
    Detector& m_det;
    unsigned m_nodeId;
    Pds::Eb::TebCtrbParams m_tPrms;
    Pds::Eb::MebCtrbParams m_mPrms;
    std::unique_ptr<Pds::Eb::TebContributor> m_tebContributor;
    std::unique_ptr<Pds::Eb::MebContributor> m_mebContributor;
    std::unique_ptr<TebReceiverBase> m_tebReceiver;
    std::unique_ptr<prometheus::Exposer> m_exposer;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    ZmqSocket m_inprocSend;
    nlohmann::json m_connectMsg;
    size_t m_collectionId;
    Pds::Trg::Factory<Pds::Trg::TriggerPrimitive> m_trigPrimFactory;
    Pds::Trg::TriggerPrimitive* m_triggerPrimitive;
    unsigned m_numTebBuffers;
    unsigned m_xpmPort;
    std::shared_ptr<PV> m_deadtimePv;
    std::string m_supervisorIpPort;
    bool m_isSupervisor;
};

}

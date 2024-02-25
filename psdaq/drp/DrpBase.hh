#pragma once

#include "drp.hh"
#include "FileWriter.hh"
#include "Detector.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/trigger/utilities.hh"
#include "psdaq/service/json.hpp"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/MebContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"
#include "psdaq/eb/ResultDgram.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/TransitionId.hh"

namespace Pds {
    class TimingHeader;
};

namespace Drp {

static const char* const RED_ON  = "\033[0;31m";
static const char* const RED_OFF = "\033[0m";


struct RunInfo
{
    std::string experimentName;
    uint32_t runNumber;
};

struct ChunkInfo
{
    std::string filename;
    uint32_t chunkId;
};

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
    FileParameters() {
        m_outputDir = "777";
        m_instrument = "777";
        m_runNumber = 777;
        m_experimentName = "777";
        m_hostname = "777";
        m_nodeId = 777;
        m_chunkId = 777;
    }

    FileParameters(const Parameters& para, const RunInfo& runInfo, std::string hostname, unsigned nodeId) {
        m_outputDir = para.outputDir;
        m_instrument = para.instrument;
        m_runNumber = runInfo.runNumber;
        m_experimentName = runInfo.experimentName;
        m_hostname = hostname;
        m_nodeId = nodeId;
        m_chunkId = 0;
    }

    bool advanceChunkId()           { ++ m_chunkId; return true; }
    // getters
    std::string outputDir()         { return m_outputDir; }
    std::string instrument()        { return m_instrument; }
    unsigned runNumber()            { return m_runNumber; }
    std::string experimentName()    { return m_experimentName; }
    std::string hostname()          { return m_hostname; }
    unsigned nodeId()               { return m_nodeId; }
    unsigned chunkId()              { return m_chunkId; }
    std::string runName();
};

class PgpReader;

class EbReceiver : public Pds::Eb::EbCtrbInBase
{
public:
    EbReceiver(Parameters& para, Pds::Eb::TebCtrbParams& tPrms, MemPool& pool,
               ZmqSocket& inprocSend, Pds::Eb::MebContributor& mon,
               const std::shared_ptr<Pds::MetricExporter>& exporter);
    void process(const Pds::Eb::ResultDgram& result, unsigned index) override;
public:
    void detector(Detector* det) {m_det = det;}
    void tsId(unsigned nodeId) {m_tsId = nodeId;}
    void resetCounters(bool all);
    void configure(Detector*, const PgpReader*);
    std::string openFiles(const Parameters& para, const RunInfo& runInfo, std::string hostname, unsigned nodeId);
    bool advanceChunkId();
    std::string reopenFiles();
    std::string closeFiles();
    uint64_t chunkSize();
    bool chunkPending();
    void chunkRequestSet();
    void chunkReset();
    bool writing();
    static const uint64_t DefaultChunkThresh = 500ull * 1024ull * 1024ull * 1024ull;    // 500 GB
    FileParameters *fileParameters()    { return &m_fileParameters; }
private:
    void _writeDgram(XtcData::Dgram* dgram);
private:
    MemPool& m_pool;
    Detector* m_det;
    const PgpReader* m_pgp;
    unsigned m_tsId;
    Pds::Eb::MebContributor& m_mon;
    BufferedFileWriterMT m_fileWriter;
    SmdWriter m_smdWriter;
    bool m_writing;
    ZmqSocket& m_inprocSend;
    uint32_t m_lastIndex;
    uint32_t m_lastEvtCounter;
    uint64_t m_lastPid;
    XtcData::TransitionId::Value m_lastTid;
    uint64_t m_offset;
    uint64_t m_chunkOffset;
    bool m_chunkRequest;
    bool m_chunkPending;
    std::vector<uint8_t> m_configureBuffer;
    uint64_t m_damage;
    uint64_t m_evtSize;
    uint64_t m_latPid;
    int64_t m_latency;
    std::shared_ptr<Pds::PromHistogram> m_dmgType;
    FileParameters m_fileParameters;
    unsigned m_partition;
};

class PgpReader
{
public:
    PgpReader(const Parameters& para, MemPool& pool, unsigned maxRetCnt, unsigned dmaFreeCnt);
    virtual ~PgpReader() {};
    int32_t read();
    void flush();
    const Pds::TimingHeader* handle(Detector* det, unsigned current);
    void freeDma(PGPEvent* event);
    virtual void handleBrokenEvent(const PGPEvent& event) {}
    virtual void resetEventCounter() { m_lastComplete = 0; } // EvtCounter reset
    const uint64_t dmaBytes()     const { return m_dmaBytes; }
    const uint64_t dmaSize()      const { return m_dmaSize; }
    const int64_t  latency()      const { return m_latency; }
    const uint64_t nDmaErrors()   const { return m_nDmaErrors; }
    const uint64_t nNoComRoG()    const { return m_nNoComRoG; }
    const uint64_t nMissingRoGs() const { return m_nMissingRoGs; }
    const uint64_t nTmgHdrError() const { return m_nTmgHdrError; }
    const uint64_t nPgpJumps()    const { return m_nPgpJumps; }
    const uint64_t nNoTrDgrams()  const { return m_nNoTrDgrams; }
    std::chrono::nanoseconds age(const XtcData::TimeStamp& time) const;
private:
    void _setTimeOffset(const XtcData::TimeStamp& time);
protected:
    const Parameters& m_para;
    MemPool& m_pool;
    pollfd m_pfd;
    Pds::fast_monotonic_clock::time_point m_t0;
    int m_tmo;
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
    std::chrono::nanoseconds m_tOffset;
};

class PV;

class DrpBase
{
public:
    DrpBase(Parameters& para, ZmqContext& context);
    void shutdown();
    nlohmann::json connectionInfo(const std::string& ip);
    std::string connect(const nlohmann::json& msg, size_t id);
    std::string configure(const nlohmann::json& msg);
    std::string beginrun(const nlohmann::json& phase1Info, RunInfo& runInfo);
    std::string endrun(const nlohmann::json& phase1Info);
    std::string enable(const nlohmann::json& phase1Info, bool& chunkRequest, ChunkInfo& chunkInfo);
    void unconfigure();
    void disconnect();
    void runInfoSupport  (XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesLookup& namesLookup);
    void runInfoData     (XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesLookup& namesLookup, const RunInfo& runInfo);
    void chunkInfoSupport(XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesLookup& namesLookup);
    void chunkInfoData   (XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesLookup& namesLookup, const ChunkInfo& chunkInfo);
    Pds::Eb::TebContributor& tebContributor() {return *m_tebContributor;}
    EbReceiver& ebReceiver() {return *m_ebRecv;}
    Pds::Trg::TriggerPrimitive* triggerPrimitive() const {return m_triggerPrimitive;}
    prometheus::Exposer* exposer() {return m_exposer.get();}
    unsigned nodeId() const {return m_nodeId;}
    const Pds::Eb::TebCtrbParams& tebPrms() const {return m_tPrms;}
    bool isSupervisor() const {return m_isSupervisor;}
    const std::string& supervisorIpPort() const {return m_supervisorIpPort;}
    MemPool pool;
private:
    int setupTriggerPrimitives(const nlohmann::json& body);
    int parseConnectionParams(const nlohmann::json& body, size_t id);
    void printParams() const;
    Parameters& m_para;
    unsigned m_nodeId;
    Pds::Eb::TebCtrbParams m_tPrms;
    Pds::Eb::MebCtrbParams m_mPrms;
    std::unique_ptr<Pds::Eb::TebContributor> m_tebContributor;
    std::unique_ptr<Pds::Eb::MebContributor> m_mebContributor;
    std::unique_ptr<EbReceiver> m_ebRecv;
    std::unique_ptr<prometheus::Exposer> m_exposer;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    ZmqSocket m_inprocSend;
    nlohmann::json m_connectMsg;
    size_t m_collectionId;
    Pds::Trg::Factory<Pds::Trg::TriggerPrimitive> m_trigPrimFactory;
    Pds::Trg::TriggerPrimitive* m_triggerPrimitive;
    std::string m_hostname;
    unsigned m_numTebBuffers;
    unsigned m_xpmPort;
    std::shared_ptr<PV> m_deadtimePv;
    std::string m_supervisorIpPort;
    bool m_isSupervisor;
};

}

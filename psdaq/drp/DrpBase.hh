#pragma once

#include "drp.hh"
#include "FileWriter.hh"
#include "Detector.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/trigger/utilities.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/MebContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"
#include "psdaq/eb/ResultDgram.hh"
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

    bool advanceChunkId()                { ++m_chunkId; return true; }
    // getters
    const std::string& outputDir()       const { return m_outputDir; }
    const std::string& instrument()      const { return m_instrument; }
    unsigned runNumber()                 const { return m_runNumber; }
    const std::string& experimentName()  const { return m_experimentName; }
    const std::string& hostname()        const { return m_hostname; }
    unsigned nodeId()                    const { return m_nodeId; }
    unsigned chunkId()                   const { return m_chunkId; }
    std::string runName() const;
};

class PgpReader;
class DrpBase;

class TebReceiverBase : public Pds::Eb::EbCtrbInBase
{
public:
  TebReceiverBase(const Parameters&, DrpBase&);
    virtual ~TebReceiverBase() {}
protected:
    void process(const Pds::Eb::ResultDgram&, unsigned index) override;
public:
    void resetCounters(bool all);
    int  connect(const std::shared_ptr<Pds::MetricExporter>);
    void unconfigure();
    std::string openFiles(const RunInfo& runInfo);
    bool advanceChunkId();
    std::string reopenFiles();
    std::string closeFiles();
    uint64_t chunkSize() const { return m_offset - m_chunkOffset; }
    void chunkRequestSet();
    void chunkReset();
    void offsetReset() { m_offset = 0; }
    void offsetAppend(size_t size) { m_offset += size; }
    bool writing() const { return m_writing; }
    static const uint64_t DefaultChunkThresh = 500ull * 1024ull * 1024ull * 1024ull;    // 500 GB
    const FileParameters& fileParameters() const { return *m_fileParameters; }
    virtual FileWriterBase& fileWriter() = 0;
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
private:
    unsigned m_tsId;
    bool m_writing;
    ZmqSocket& m_inprocSend;
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

class PgpReader
{
public:
    PgpReader(const Parameters& para, MemPool& pool, unsigned maxRetCnt, unsigned dmaFreeCnt);
    virtual ~PgpReader();
    int32_t read();
    void flush();
    const Pds::TimingHeader* handle(Detector* det, unsigned current);
    void freeDma(PGPEvent* event);
    virtual void handleBrokenEvent(const PGPEvent& event) {}
    virtual void resetEventCounter() { m_lastComplete = 0; } // EvtCounter reset
    uint64_t dmaBytes()     const { return m_dmaBytes; }
    uint64_t dmaSize()      const { return m_dmaSize; }
    int64_t  latency()      const { return m_latency; }
    uint64_t nDmaErrors()   const { return m_nDmaErrors; }
    uint64_t nNoComRoG()    const { return m_nNoComRoG; }
    uint64_t nMissingRoGs() const { return m_nMissingRoGs; }
    uint64_t nTmgHdrError() const { return m_nTmgHdrError; }
    uint64_t nPgpJumps()    const { return m_nPgpJumps; }
    uint64_t nNoTrDgrams()  const { return m_nNoTrDgrams; }
    int64_t  nPgpInUser()   const { return dmaGetRxBuffinUserCount  (m_pool.fd()); }
    int64_t  nPgpInHw()     const { return dmaGetRxBuffinHwCount    (m_pool.fd()); }
    int64_t  nPgpInPreHw()  const { return dmaGetRxBuffinPreHwQCount(m_pool.fd()); }
    int64_t  nPgpInRx()     const { return dmaGetRxBuffinSwQCount   (m_pool.fd()); }
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
    unsigned m_dmaRetCnt;
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

class DrpBase
{
public:
    DrpBase(Parameters& para, MemPool& pool, Detector& det, ZmqContext& context);
protected:
    void setTebReceiver(std::unique_ptr<TebReceiverBase> tebRecv) { m_tebReceiver = std::move(tebRecv); }
public:
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
    Detector& detector() const {return m_det; }
    Pds::Eb::TebContributor& tebContributor() const {return *m_tebContributor;}
    Pds::Eb::MebContributor& mebContributor() const {return *m_mebContributor;}
    TebReceiverBase& tebReceiver() const {return *m_tebReceiver;}
    Pds::Trg::TriggerPrimitive* triggerPrimitive() const {return m_triggerPrimitive;}
    prometheus::Exposer* exposer() const { return m_exposer.get(); }
    ZmqSocket& inprocSend() { return m_inprocSend; }
    unsigned nodeId() const {return m_nodeId;}
    const Pds::Eb::TebCtrbParams& tebPrms() const {return m_tPrms;}
    bool isSupervisor() const {return m_isSupervisor;}
    const std::string& supervisorIpPort() const {return m_supervisorIpPort;}
    MemPool& pool;
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

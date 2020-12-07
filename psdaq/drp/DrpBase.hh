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
#include "xtcdata/xtc/NamesLookup.hh"

namespace Drp {

static const char* const RED_ON  = "\033[0;31m";
static const char* const RED_OFF = "\033[0m";


struct RunInfo
{
    std::string experimentName;
    uint32_t runNumber;
};

class EbReceiver : public Pds::Eb::EbCtrbInBase
{
public:
    EbReceiver(const Parameters& para, Pds::Eb::TebCtrbParams& tPrms, MemPool& pool,
               ZmqSocket& inprocSend, Pds::Eb::MebContributor& mon,
               const std::shared_ptr<Pds::MetricExporter>& exporter);
    void process(const Pds::Eb::ResultDgram& result, const void* input) override;
public:
    void resetCounters();
    std::string openFiles(const Parameters& para, const RunInfo& runInfo, std::string hostname, unsigned nodeId);
    std::string closeFiles();
private:
    void _writeDgram(XtcData::Dgram* dgram);
private:
    MemPool& m_pool;
    Pds::Eb::MebContributor& m_mon;
    BufferedFileWriterMT m_fileWriter;
    SmdWriter m_smdWriter;
    bool m_writing;
    ZmqSocket& m_inprocSend;
    static const int m_size = 100;
    uint32_t m_indices[m_size];
    int m_count;
    uint32_t m_lastIndex;
    uint32_t m_lastEvtCounter;
    uint64_t m_lastPid;
    XtcData::TransitionId::Value m_lastTid;
    uint64_t m_offset;
    std::vector<uint8_t> m_configureBuffer;
    uint64_t m_damage;
    std::shared_ptr<Pds::PromHistogram> m_dmgType;

};

class DrpBase
{
public:
    DrpBase(Parameters& para, ZmqContext& context);
    void shutdown();
    void reset();
    nlohmann::json connectionInfo(const std::string& ip);
    std::string connect(const nlohmann::json& msg, size_t id);
    std::string configure(const nlohmann::json& msg);
    std::string beginrun(const nlohmann::json& phase1Info, RunInfo& runInfo);
    std::string endrun(const nlohmann::json& phase1Info);
    void unconfigure();
    void disconnect();
    void runInfoSupport(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup);
    void runInfoData(XtcData::Xtc& xtc, XtcData::NamesLookup& namesLookup, const RunInfo& runInfo);
    Pds::Eb::TebContributor& tebContributor() {return *m_tebContributor;}
    Pds::Trg::TriggerPrimitive* triggerPrimitive() const {return m_triggerPrimitive;}
    prometheus::Exposer* exposer() {return m_exposer.get();}
    unsigned nodeId() const {return m_nodeId;}
    const Pds::Eb::TebCtrbParams& tebPrms() const {return m_tPrms;}
    void stop() { m_tebContributor->stop(); }
    MemPool pool;
private:
    int setupTriggerPrimitives(const nlohmann::json& body);
    void parseConnectionParams(const nlohmann::json& body, size_t id);
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
};

}

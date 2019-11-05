#pragma once

#include "drp.hh"
#include "FileWriter.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/trigger/utilities.hh"
#include "psdaq/service/json.hpp"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/MebContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"
#include "psdaq/eb/ResultDgram.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"

namespace Drp {

static const char* const RED_ON  = "\033[0;31m";
static const char* const RED_OFF = "\033[0m";

class EbReceiver : public Pds::Eb::EbCtrbInBase
{
public:
    EbReceiver(const Parameters& para, Pds::Eb::TebCtrbParams& tPrms, MemPool& pool,
               ZmqSocket& inprocSend, Pds::Eb::MebContributor* mon,
               const std::shared_ptr<MetricExporter>& exporter);
    void process(const Pds::Eb::ResultDgram& result, const void* input) override;
public:
    void resetCounters();
private:
    MemPool& m_pool;
    Pds::Eb::MebContributor* m_mon;
    BufferedFileWriter m_fileWriter;
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
    unsigned m_nodeId;
};

class DrpBase
{
public:
    DrpBase(Parameters& para, ZmqContext& context);
    void shutdown();
    nlohmann::json connectionInfo();
    std::string connect(const nlohmann::json& msg, size_t id);
    std::string configure(const nlohmann::json& msg);
    std::string beginrun(const nlohmann::json& msg);
    std::string endrun(const nlohmann::json& msg);
    Pds::Eb::TebContributor& tebContributor() const {return *m_tebContributor;}
    Pds::Trg::TriggerPrimitive* triggerPrimitive() const {return m_triggerPrimitive;}
    prometheus::Exposer* exposer() {return m_exposer.get();}
    unsigned nodeId() const {return m_nodeId;}
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
    std::unique_ptr<Pds::Eb::MebContributor> m_meb;
    std::unique_ptr<EbReceiver> m_ebRecv;
    std::unique_ptr<prometheus::Exposer> m_exposer;
    std::shared_ptr<MetricExporter> m_exporter;
    ZmqSocket m_inprocSend;
    nlohmann::json m_connectMsg;
    size_t m_collectionId;
    Pds::Trg::Factory<Pds::Trg::TriggerPrimitive> m_trigPrimFactory;
    Pds::Trg::TriggerPrimitive* m_triggerPrimitive;
};

}

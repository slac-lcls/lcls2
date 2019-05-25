#pragma once

#include <thread>
#include "PGPReader.hh"
#include "FileWriter.hh"
#include "psdaq/eb/eb.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/MebContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"

class MetricExporter;

namespace Drp {

class Detector;

#pragma pack(push, 4)
class MyDgram : public XtcData::Dgram {
public:
    MyDgram(XtcData::Dgram& dgram, uint64_t val, unsigned contributor_id);
private:
    uint64_t _data;
};
#pragma pack(pop)

class EbReceiver : public Pds::Eb::EbCtrbInBase
{
public:
    EbReceiver(const Parameters& para, Pds::Eb::TebCtrbParams& tPrms, MemPool& pool,
               ZmqContext& context, Pds::Eb::MebContributor* mon,
               std::shared_ptr<MetricExporter> exporter);
    virtual ~EbReceiver() {};
    void process(const XtcData::Dgram* result, const void* input) override;
private:
    MemPool& m_pool;
    Pds::Eb::MebContributor* m_mon;
    BufferedFileWriter m_fileWriter;
    SmdWriter m_smdWriter;
    bool m_writing;
    ZmqSocket m_inprocSend;
    static const int m_size = 100;
    uint32_t m_indices[m_size];
    int m_count;
    uint32_t lastIndex;
    uint32_t lastEvtCounter;
    uint64_t m_offset;
    unsigned m_nodeId;
};

struct Parameters;
struct MemPool;

class DrpApp : public CollectionApp
{
public:
    DrpApp(Parameters* para);
    nlohmann::json connectionInfo() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void handleReset(const nlohmann::json& msg) override;
    void shutdown();
private:
    void parseConnectionParams(const nlohmann::json& msg);
    void collector();

    Parameters* m_para;
    Pds::Eb::TebCtrbParams m_tPrms;
    Pds::Eb::MebCtrbParams m_mPrms;
    std::thread m_pgpThread;
    std::thread m_collectorThread;
    MemPool m_pool;
    std::unique_ptr<PGPReader> m_pgpReader;
    std::unique_ptr<Pds::Eb::TebContributor> m_ebContributor;
    std::unique_ptr<EbReceiver> m_ebRecv;
    std::unique_ptr<Pds::Eb::MebContributor> m_meb;
    std::unique_ptr<prometheus::Exposer> m_exposer;
    Detector* m_det;
};

}

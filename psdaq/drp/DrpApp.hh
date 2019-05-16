#pragma once

#include <thread>
#include "PGPReader.hh"
#include "psdaq/eb/eb.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/MebContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"
#include "psdaq/eb/StatsMonitor.hh"

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

class BufferedFileWriter
{
public:
    BufferedFileWriter();
    ~BufferedFileWriter();
    void open(std::string& fileName);
    void writeEvent(void* data, size_t size);
private:
    int m_fd;
    int m_count;
    std::vector<uint8_t> m_buffer;
    // 4 MB
    static const size_t BufferSize = 4194304;
};


class EbReceiver : public Pds::Eb::EbCtrbInBase
{
public:
    EbReceiver(const Parameters& para, Pds::Eb::TebCtrbParams& tPrms, MemPool& pool,
               ZmqContext& context, Pds::Eb::MebContributor* mon,
               Pds::Eb::StatsMonitor& smon);
    virtual ~EbReceiver() {};
    void process(const XtcData::Dgram* result, const void* input) override;
private:
    MemPool& m_pool;
    Pds::Eb::MebContributor* m_mon;
    BufferedFileWriter m_fileWriter;
    bool m_writing;
    ZmqSocket m_inprocSend;
    static const int m_size = 100;
    uint32_t m_indices[m_size];
    int m_count;
    uint32_t lastIndex;
    uint32_t lastEvtCounter;
};

struct Parameters;
struct MemPool;

class DrpApp : public CollectionApp
{
public:
    DrpApp(Parameters* para);
    json connectionInfo() override;
    void handleConnect(const json& msg) override;
    void handlePhase1(const json& msg) override;
    void handleReset(const json& msg) override;
    void shutdown();
private:
    void parseConnectionParams(const json& msg);
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
    prometheus::Exposer m_exposer;
    Pds::Eb::StatsMonitor m_smon;
    Detector* m_det;
};

}

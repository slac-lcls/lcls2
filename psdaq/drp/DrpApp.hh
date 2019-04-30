#pragma once

#include <thread>
//#include "Collector.hh"
#include "PGPReader.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/MebContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"
#include "psdaq/eb/StatsMonitor.hh"

class Detector;

#pragma pack(push, 4)
class MyDgram : public XtcData::Dgram {
public:
    MyDgram(XtcData::Dgram& dgram, uint64_t val, unsigned contributor_id);
private:
    uint64_t _data;
};
#pragma pack(pop)

// return dma indices in batches for performance
class DmaIndexReturner
{
public:
    DmaIndexReturner(int fd);
    ~DmaIndexReturner();
    void returnIndex(uint32_t index);
private:
    static const int BatchSize = 500;
    int m_fd;
    int m_counts;
    uint32_t m_indices[BatchSize];
};

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
    EbReceiver(const Parameters& para, MemPool& pool,
               ZmqContext& context, Pds::Eb::MebContributor* mon,
               Pds::Eb::StatsMonitor& smon);
    virtual ~EbReceiver() {};
    void process(const XtcData::Dgram* result, const void* input) override;
private:
    MemPool& m_pool;
    Pds::Eb::MebContributor* m_mon;
    unsigned nreceive;
    BufferedFileWriter m_fileWriter;
    bool m_writing;
    DmaIndexReturner m_indexReturner;
    ZmqSocket m_inprocSend;
};

struct Parameters;
struct MemPool;

class DrpApp : public CollectionApp
{
public:
    DrpApp(Parameters* para);
    void handleConnect(const json& msg) override;
    void handlePhase1(const json& msg) override;
    void handleReset(const json& msg) override;
private:
    void parseConnectionParams(const json& msg);
    void collector();

    Parameters* m_para;
    std::thread m_pgpThread;
    std::thread m_collectorThread;
    std::thread m_monitorThread;
    MemPool m_pool;
    std::unique_ptr<PGPReader> m_pgpReader;
    std::unique_ptr<Pds::Eb::TebContributor> m_ebContributor;
    std::unique_ptr<EbReceiver> m_ebRecv;
    std::unique_ptr<Pds::Eb::MebContributor> m_meb;
    Pds::Eb::StatsMonitor m_smon;
    Detector* m_det;
};

#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include "Detector.hh"
#include "drp.hh"
#include "spscqueue.hh"

namespace Pds {
    class MetricExporter;
    namespace Eb { class TebContributor;}
};

namespace Drp {

struct Batch
{
    uint32_t start;
    uint32_t size;
};

class PGPDrp;

class Pgp: public PgpReader
{
public:
    Pgp(const Parameters& para, MemPool& pool, Detector& det, PGPDrp& m_drp);
    virtual void handleBrokenEvent(const PGPEvent& event) override;
    virtual void resetEventCounter() override;
private:
    static const int MAX_RET_CNT_C = 1000;
    PGPDrp& m_drp;
};

class PGPDrp : public DrpBase
{
public:
    PGPDrp(Parameters&, MemPool&, Detector&, ZmqContext&,
           int* inpMqId, int* resMqId, int* inpShmId, int* resShmId, size_t shemeSize);
    void reader();
    void collector();
    void handleBrokenEvent(const PGPEvent& event);
    void resetEventCounter();
    std::string configure(const nlohmann::json& msg);
    unsigned unconfigure();
public:
    const PgpReader* pgp() const { return &m_pgp; }
private:
    int  _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
private:
    const Parameters& m_para;
    Detector& m_det;
    Pgp m_pgp;
    std::vector<SPSCQueue<Batch> > m_workerInputQueues;
    std::vector<SPSCQueue<Batch> > m_workerOutputQueues;
    std::thread m_pgpThread;
    std::vector<std::thread> m_workerThreads;
    std::thread m_collectorThread;
    std::atomic<bool> m_terminate;
    uint64_t m_nDmaRet;
    uint64_t m_nevents;
    Batch m_batch;
    int* m_inpMqId;
    int* m_resMqId;
    int* m_inpShmId;
    int* m_resShmId;
    std::atomic<int> threadCountWrite;
    std::atomic<int> threadCountPush;
    unsigned m_flushTmo;
    size_t m_shmemSize;
    int64_t m_pyAppTime;
    bool m_pythonDrp;
};

}

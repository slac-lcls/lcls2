#pragma once

#include <array>
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
    uint64_t l1count;
};

class PGPDrp;

class Pgp: public PgpReader
{
public:
    Pgp(const Parameters& para, MemPool& pool, Detector& det, PGPDrp& m_drp);
    virtual void handleBrokenEvent(const PGPEvent& event) override;
    virtual void resetEventCounter() override;
private:
    static const unsigned MAX_RET_CNT_C = 1000;
    PGPDrp& m_drp;
};

class PGPDrp : public DrpBase
{
public:
    PGPDrp(Parameters&, MemPool&, Detector&, ZmqContext&,
           int* inpMqId, int* resMqId, int* inpShmId, int* resShmId, size_t shemeSize);
    virtual ~PGPDrp() {}
    void reader();
    void collector();
    void handleBrokenEvent(const PGPEvent& event);
    void resetEventCounter();
    std::string configure(const nlohmann::json& msg);
    unsigned unconfigure();
    std::string startup(XtcData::Xtc& xtc, const void* be);
private:
    int  _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
    void _reportStale(uint32_t staleEvtCtr) const;
private:
    static const unsigned SkipRing = 64;
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
    uint64_t m_nBatchGaps;                                   // Reader batches not ending at the current evtCounter
    std::atomic<uint32_t> m_collectorEvtCtr;                 // Last evtCounter the Collector reached
    std::atomic<uint64_t> m_nSkips;                          // Events the Collector skipped as broken
    std::array<std::atomic<uint32_t>, SkipRing> m_skipped;   // evtCounters of the most recent skips
};

}

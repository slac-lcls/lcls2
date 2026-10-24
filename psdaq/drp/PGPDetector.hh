/**
 * @file
 * @brief PGPDrp, the DRP for PGP-card detectors: a reader thread groups DMA events into batches for worker threads, and a collector passes the results to the TEB contributor.
 */
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

/** A run of consecutive events (by event counter) handed to one worker. */
struct Batch
{
    uint32_t start;  ///< Event counter of the first event in the batch.
    uint32_t size;  ///< Number of events in the batch, including broken events (see PGPDrp::handleBrokenEvent()).
    uint64_t l1count;  ///< Number of L1Accepts counted up to the start of this batch.
};

class PGPDrp;

/** PgpReader for the PGP-card DRP that forwards broken-event and event-counter-reset notifications to its PGPDrp. */
class Pgp: public PgpReader
{
public:
    /** Construct the PgpReader (at most 1000 buffers per read, 32 per free batch) and set the driver mask for laneMask on the virtual channel of det; aborts with a message if that fails. */
    Pgp(const Parameters& para, MemPool& pool, Detector& det, PGPDrp& m_drp);
    /** Forward event to PGPDrp::handleBrokenEvent, which counts it in the current batch. */
    virtual void handleBrokenEvent(const PGPEvent& event) override;
    /** Call PgpReader::resetEventCounter (last completed event counter to 0), then PGPDrp::resetEventCounter (batching restarts at event counter 1). */
    virtual void resetEventCounter() override;
private:
    static const unsigned MAX_RET_CNT_C = 1000;
    PGPDrp& m_drp;
};

/** DrpBase for PGP-card detectors: reader() groups DMA events into batches that are processed by nworkers worker threads (optionally through a Python DRP over message queues and shared memory), and collector() frees the DMA buffers and passes each event to the TEB contributor. */
class PGPDrp : public DrpBase
{
public:
    /** Construct DrpBase and the PGP reader, keep the message-queue and shared-memory IDs for the workers, set the flush timeout from the batch size, and install a TebReceiver (or a CubeTebReceiver when cube workers are configured). */
    PGPDrp(Parameters&, MemPool&, Detector&, ZmqContext&,
           int* inpMqId, int* resMqId, int* inpShmId, int* resShmId, size_t shemeSize);
    /** Does nothing; empty body. */
    virtual ~PGPDrp() {}
    /** Reader thread body: read DMA buffers, assemble events and cut batches at batch-size pulse ID boundaries or transitions (other than SlowUpdate), pushing them to the workers in turn; a partial batch is pushed after the flush timeout without data. Transitions are also copied into their transition buffer and the TEB input buffer, and are given to every worker when the Python DRP is used. */
    void reader();
    /** Collector thread body: take batches from the worker output queues in turn, free the DMA buffers of each event and pass its pebble index to TebContributor::process(); while waiting, time out TEB batches. */
    void collector();
    /** Count a broken event in the current batch, since the firmware advanced the event counter (per the code comment). */
    void handleBrokenEvent(const PGPEvent& event);
    /** Restart batching at event counter 1 with an empty batch and an L1Accept count of 0. */
    void resetEventCounter();
    /** Call DrpBase::configure(), decide from the drp kwarg whether the Python DRP is used (then the pythonScript kwarg must be set and at most 511 characters), and create one input and one output queue per worker. Returns an empty string or an error message. */
    std::string configure(const nlohmann::json& msg);
    /** Call DrpBase::unconfigure(), stop and join the worker, reader and collector threads, and drop the worker queues; returns 0. */
    unsigned unconfigure();
    /** Call DrpBase::startup() and start the worker threads, the reader thread and the collector thread; returns an empty string or the DrpBase error. */
    std::string startup(XtcData::Xtc& xtc, const void* be);
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

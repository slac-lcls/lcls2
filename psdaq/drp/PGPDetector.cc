#include <iostream>
#include <fstream>
#include <limits.h>
#include "DataDriver.h"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/eb/TebContributor.hh"
#include "DrpBase.hh"
#include "PGPDetector.hh"

using logging = psalg::SysLog;

namespace Drp {

long readInfinibandCounter(const std::string& counter)
{
    std::string path{"/sys/class/infiniband/mlx5_0/ports/1/counters/" + counter};
    std::ifstream in(path);
    if (in.is_open()) {
        std::string line;
        std::getline(in, line);
        return stol(line);
    }
    else {
        return 0;
    }
}

bool checkPulseIds(MemPool& pool, PGPEvent* event)
{
    uint64_t pulseId = 0;
    for (int i=0; i<4; i++) {
        if (event->mask & (1 << i)) {
            uint32_t index = event->buffers[i].index;
            const Pds::TimingHeader* timingHeader = reinterpret_cast<Pds::TimingHeader*>(pool.dmaBuffers[index]);
            if (pulseId == 0) {
                pulseId = timingHeader->pulseId();
            }
            else {
                if (pulseId != timingHeader->pulseId()) {
                    logging::error("Wrong pulse id! expected %lu but got %lu instead",
                           pulseId, timingHeader->pulseId());
                    return false;
                }
            }
            // check bit 7 in pulseId for error
            bool error = timingHeader->control() & (1 << 7);
            if (error) {
                logging::error("Error bit in pulseId is set");
            }
        }
    }
    return true;
}

void workerFunc(const Parameters& para, DrpBase& drp, Detector* det,
                SPSCQueue<Batch>& inputQueue, SPSCQueue<Batch>& outputQueue)
{
    Batch batch;
    MemPool& pool = drp.pool;
    const unsigned nbuffers = pool.nbuffers();
    while (true) {
        if (!inputQueue.pop(batch)) {
            break;
        }

        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) % nbuffers;
            PGPEvent* event = &pool.pgpEvents[index];
            checkPulseIds(pool, event);

            Pds::EbDgram* dgram = reinterpret_cast<Pds::EbDgram*>(pool.pebble[index]);
            XtcData::TransitionId::Value transitionId = dgram->service();

            // Event
            if (transitionId == XtcData::TransitionId::L1Accept) {
                det->event(*dgram, event);
                // make sure the detector hasn't made the event too big
                if (dgram->xtc.extent > pool.bufferSize()) {
                    logging::critical("L1Accept: buffer size (%d) too small for requested extent (%d)", pool.bufferSize(), dgram->xtc.extent);
                    exit(-1);
                }

                if (event->l3InpBuf) {  // else timed out
                    Pds::EbDgram* l3InpDg = new(event->l3InpBuf) Pds::EbDgram(*dgram);
                    if (drp.triggerPrimitive()) { // else this DRP doesn't provide input
                        drp.triggerPrimitive()->event(pool, index, dgram->xtc, l3InpDg->xtc);
                        size_t size = sizeof(*l3InpDg) + l3InpDg->xtc.sizeofPayload();
                        if (size > drp.tebPrms().maxInputSize) {
                            logging::critical("L3 Input Dgram of size %zd overflowed buffer of size %zd", size, drp.tebPrms().maxInputSize);
                            exit(-1);
                        }
                    }
                }
            }
            // transitions
            else {
                // Since the Transition Dgram's XTC was already created on
                // phase1 of the transition, fix up the Dgram header with the
                // real one while taking care not to touch the XTC
                // Revisit: Delay this until EbReceiver time?
                Pds::EbDgram* trDgram = pool.transitionDgram();
                memcpy(trDgram, dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                // make sure the detector hasn't made the transition too big
                size_t size = sizeof(*trDgram) + trDgram->xtc.sizeofPayload();
                if (size > para.maxTrSize) {
                    logging::critical("Transition: buffer size (%zd) too small for Dgram (%zd)", para.maxTrSize, size);
                    exit(-1);
                }

                if (event->l3InpBuf) { // else timed out
                    new(event->l3InpBuf) Pds::EbDgram(*dgram);
                }
            }
        }

        outputQueue.push(batch);
    }
}

PGPDetector::PGPDetector(const Parameters& para, DrpBase& drp, Detector* det) :
    m_para(para), m_pool(drp.pool), m_terminate(false)
{
    m_nodeId = det->nodeId;
    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    for (int i=0; i<4; i++) {
        if (para.laneMask & (1 << i)) {
            logging::info("setting lane  %d", i);
            dmaAddMaskBytes(mask, dmaDest(i, 0));
        }
    }
    dmaSetMaskBytes(drp.pool.fd(), mask);

    for (unsigned i=0; i<para.nworkers; i++) {
        m_workerInputQueues.emplace_back(SPSCQueue<Batch>(drp.pool.nbuffers()));
        m_workerOutputQueues.emplace_back(SPSCQueue<Batch>(drp.pool.nbuffers()));
    }


    for (unsigned i = 0; i < para.nworkers; i++) {
        m_workerThreads.emplace_back(workerFunc,
                                   std::ref(para),
                                   std::ref(drp),
                                   det,
                                   std::ref(m_workerInputQueues[i]),
                                   std::ref(m_workerOutputQueues[i]));
    }
}

void PGPDetector::reader(std::shared_ptr<MetricExporter> exporter,
                         Pds::Eb::TebContributor& tebContributor)
{
    // setup monitoring
    uint64_t nevents = 0L;
    uint64_t bytes = 0L;
    std::map<std::string, std::string> labels{{"partition", std::to_string(m_para.partition)}};
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return nevents;});

    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return bytes;});

    exporter->add("drp_port_rcv_rate", labels, MetricType::Rate,
                  [](){return 4*readInfinibandCounter("port_rcv_data");});

    exporter->add("drp_port_xmit_rate", labels, MetricType::Rate,
                  [](){return 4*readInfinibandCounter("port_xmit_data");});

    auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec) {
        size_t sum = 0;
        for (auto& q: vec) {
            sum += q.guess_size();
        }
        return sum;
    };
    exporter->add("drp_worker_input_queue", labels, MetricType::Gauge,
                  [&](){return queueLength(m_workerInputQueues);});

    exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
                   [&](){return queueLength(m_workerOutputQueues);});


    int64_t worker = 0L;
    uint64_t batchId = 0L;
    const unsigned bufferMask = m_pool.nbuffers() - 1;
    XtcData::TransitionId::Value lastTid;
    uint32_t lastData[6];
    while (1) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        int32_t ret = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
        for (int b=0; b < ret; b++) {
            uint32_t size = dmaRet[b];
            uint32_t index = dmaIndex[b];
            uint32_t lane = (dest[b] >> 8) & 7;
            bytes += size;
            if (unsigned(size) > m_pool.dmaSize()) {
                logging::critical("DMA overflowed buffer: %d vs %d", size, m_pool.dmaSize());
                exit(-1);
            }

            uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
            uint32_t evtCounter = data[5] & 0xffffff;
            uint32_t current = evtCounter & bufferMask;
            PGPEvent* event = &m_pool.pgpEvents[current];

            DmaBuffer* buffer = &event->buffers[lane];
            buffer->size = size;
            buffer->index = index;
            event->mask |= (1 << lane);

            if (event->mask == m_para.laneMask) {
                const Pds::TimingHeader* timingHeader = reinterpret_cast<Pds::TimingHeader*>(data);
                XtcData::TransitionId::Value transitionId = timingHeader->service();
                if (transitionId != XtcData::TransitionId::L1Accept) {
                    logging::debug("PGPReader  saw %s transition @ %d.%09d (%014lx)",
                                   XtcData::TransitionId::name(transitionId),
                                   timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                                   timingHeader->pulseId());
                }
                if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
                    logging::critical("%sFatal: Jump in complete l1Count %u -> %u | difference %d, tid %s%s",
                           RED_ON, m_lastComplete, evtCounter, evtCounter - m_lastComplete, XtcData::TransitionId::name(transitionId), RED_OFF);
                    logging::critical("data: %08x %08x %08x %08x %08x %08x",
                           data[0], data[1], data[2], data[3], data[4], data[5]);

                    logging::critical("lastTid %s", XtcData::TransitionId::name(lastTid));
                    logging::critical("lastData: %08x %08x %08x %08x %08x %08x",
                           lastData[0], lastData[1], lastData[2], lastData[3], lastData[4], lastData[5]);

                    throw "Jump in event counter";

                    for (unsigned e=m_lastComplete+1; e<evtCounter; e++) {
                        PGPEvent* brokenEvent = &m_pool.pgpEvents[e & bufferMask];
                        logging::error("broken event:  %08x", brokenEvent->mask);
                        brokenEvent->mask = 0;

                    }
                }
                m_lastComplete = evtCounter;
                lastTid = transitionId;
                memcpy(lastData, data, 24);

                nevents++;
                m_batch.size++;

                // make new dgram in the pebble
                Pds::EbDgram* dgram = new(m_pool.pebble[current]) Pds::EbDgram(*timingHeader, XtcData::Src(m_nodeId), m_para.rogMask);

                // To ensure L3 Input Dgrams appear in the batch in sequenctial
                // order, entry allocation must occur here rather than in the
                // worker threads, the excecution order of which may get scrambled
                event->l3InpBuf = tebContributor.allocate(dgram, (void*)((uintptr_t)current));

                // send batch to worker if batch is full or if it's a transition
                if (((batchId ^ timingHeader->pulseId()) & ~(m_para.batchSize - 1)) ||
                    ((transitionId != XtcData::TransitionId::L1Accept) &&
                     (transitionId != XtcData::TransitionId::SlowUpdate))) {
                    m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                    worker++;
                    m_batch.start = evtCounter + 1;
                    m_batch.size = 0;
                    batchId = timingHeader->pulseId();
                }
            }
        }
    }
}

void PGPDetector::collector(Pds::Eb::TebContributor& tebContributor)
{
    int64_t worker = 0L;
    Batch batch;
    const unsigned nbuffers = m_pool.nbuffers();
    while (true) {
        if (!m_workerOutputQueues[worker % m_para.nworkers].pop(batch)) {
            break;
        }
        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) % nbuffers;
            PGPEvent* event = &m_pool.pgpEvents[index];
            if (event->l3InpBuf) // else timed out
            {
                Pds::EbDgram* dgram = static_cast<Pds::EbDgram*>(event->l3InpBuf);
                tebContributor.process(dgram);
            }
        }
        worker++;
    }
}

void PGPDetector::resetEventCounter()
{
    m_lastComplete = 0;
    m_batch.start = 1;
    m_batch.size = 0;
}

void PGPDetector::shutdown()
{
    m_terminate.store(true, std::memory_order_release);
    logging::info("shutting down PGPReader");
    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerInputQueues[i].shutdown();
        if (m_workerThreads[i].joinable()) {
            m_workerThreads[i].join();
        }
    }
    logging::info("Worker threads finished");
    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerOutputQueues[i].shutdown();
    }
}

}

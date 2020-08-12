#include <iostream>
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
#include "EventBatcher.hh"

using logging = psalg::SysLog;

using namespace Drp;

bool checkPulseIds(const Detector* det, PGPEvent* event)
{
    uint64_t pulseId = 0;
    for (int i=0; i<4; i++) {
        if (event->mask & (1 << i)) {
            uint32_t index = event->buffers[i].index;
            const Pds::TimingHeader* timingHeader = det->getTimingHeader(index);
            if (pulseId == 0) {
                pulseId = timingHeader->pulseId();
            }
            else {
                if (pulseId != timingHeader->pulseId()) {
                    logging::error("Wrong pulse id! expected %014lx but got %014lx instead",
                                   pulseId, timingHeader->pulseId());
                    return false;
                }
            }
            if (timingHeader->error()) {
                logging::error("Timing header error bit is set");
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
            checkPulseIds(det, event);

            // get transitionId from the first lane in the event
            int lane = __builtin_ffs(event->mask) - 1;
            uint32_t dmaIndex = event->buffers[lane].index;
            const Pds::TimingHeader* timingHeader = det->getTimingHeader(dmaIndex);

            // make new dgram in the pebble
            // It must be an EbDgram in order to be able to send it to the MEB
            Pds::EbDgram* dgram = new(pool.pebble[index]) Pds::EbDgram(*timingHeader, XtcData::Src(det->nodeId), para.rogMask);
            XtcData::TransitionId::Value transitionId = dgram->service();

            // Event
            if (transitionId == XtcData::TransitionId::L1Accept) {
                det->event(*dgram, event);
                // make sure the detector hasn't made the event too big
                if (dgram->xtc.extent > pool.bufferSize()) {
                    logging::critical("L1Accept: buffer size (%d) too small for requested extent (%d)", pool.bufferSize(), dgram->xtc.extent);
                    throw "Buffer too small";
                }

                if (event->l3InpBuf) {  // else shutting down
                    Pds::EbDgram* l3InpDg = new(event->l3InpBuf) Pds::EbDgram(*dgram);
                    if (drp.triggerPrimitive()) { // else this DRP doesn't provide input
                        drp.triggerPrimitive()->event(pool, index, dgram->xtc, l3InpDg->xtc);
                        size_t size = sizeof(*l3InpDg) + l3InpDg->xtc.sizeofPayload();
                        if (size > drp.tebPrms().maxInputSize) {
                            logging::critical("L3 Input Dgram of size %zd overflowed buffer of size %zd", size, drp.tebPrms().maxInputSize);
                            throw "Input Dgram overflowed buffer";
                        }
                    }
                }
            }
            // transitions
            else {
                logging::debug("PGPDetector saw %s transition @ %u.%09u (%014lx)",
                               XtcData::TransitionId::name(transitionId),
                               dgram->time.seconds(), dgram->time.nanoseconds(), timingHeader->pulseId());
                // Allocate a transition dgram from the pool and initialize its header
                Pds::EbDgram* trDgram = pool.allocateTr();
                memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                if (transitionId != XtcData::TransitionId::SlowUpdate) {
                   // copy the temporary xtc created on phase 1 of the transition
                   // into the real location
                   XtcData::Xtc& trXtc = det->transitionXtc();
                   memcpy((void*)&trDgram->xtc, (const void*)&trXtc, trXtc.extent);
                }
                else {
                   det->slowupdate(trDgram->xtc);
                }
                // make sure the detector hasn't made the transition too big
                size_t size = sizeof(*trDgram) + trDgram->xtc.sizeofPayload();
                if (size > para.maxTrSize) {
                    logging::critical("Transition: buffer size (%zd) too small for Dgram (%zd)", para.maxTrSize, size);
                    throw "Buffer too small";
                }

                if (event->l3InpBuf) { // else shutting down
                    new(event->l3InpBuf) Pds::EbDgram(*dgram);
                }
                event->transitionDgram = trDgram;
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
            dmaAddMaskBytes(mask, dmaDest(i, det->virtChan));
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

PGPDetector::~PGPDetector()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    shutdown();
}

void PGPDetector::reader(std::shared_ptr<Pds::MetricExporter> exporter, Detector* det,
                         Pds::Eb::TebContributor& tebContributor)
{
    // setup monitoring
    uint64_t nevents = 0L;
    uint64_t bytes = 0L;
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName}};
    exporter->add("drp_event_rate", labels, Pds::MetricType::Rate,
                  [&](){return nevents;});

    exporter->add("drp_pgp_byte_rate", labels, Pds::MetricType::Rate,
                  [&](){return bytes;});

    auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec) {
        size_t sum = 0;
        for (auto& q: vec) {
            sum += q.guess_size();
        }
        return sum;
    };
    exporter->add("drp_worker_input_queue", labels, Pds::MetricType::Gauge,
                  [&](){return queueLength(m_workerInputQueues);});

    exporter->add("drp_worker_output_queue", labels, Pds::MetricType::Gauge,
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
            if (size > m_pool.dmaSize()) {
                logging::critical("DMA overflowed buffer: %u vs %u", size, m_pool.dmaSize());
                throw "DMA overflowed buffer";
            }

            const Pds::TimingHeader* timingHeader = det->getTimingHeader(index);
            uint32_t evtCounter = timingHeader->evtCounter & 0xffffff;
            uint32_t current = evtCounter & bufferMask;
            PGPEvent* event = &m_pool.pgpEvents[current];

            DmaBuffer* buffer = &event->buffers[lane];
            buffer->size = size;
            buffer->index = index;
            event->mask |= (1 << lane);

            const uint32_t* data = reinterpret_cast<const uint32_t*>(timingHeader);
            if (m_para.verbose < 2)
                logging::debug("PGPReader  lane %u  size %u  hdr %016lx.%016lx.%08x",
                               lane, size,
                               reinterpret_cast<const uint64_t*>(data)[0],
                               reinterpret_cast<const uint64_t*>(data)[1],
                               reinterpret_cast<const uint32_t*>(data)[4]);

            if (event->mask == m_para.laneMask) {
                XtcData::TransitionId::Value transitionId = timingHeader->service();
                if (transitionId != XtcData::TransitionId::L1Accept) {
                    if (transitionId==XtcData::TransitionId::Configure) {
                        logging::info("PGPReader saw %s transition @ %u.%09u (%014lx)",
                                      XtcData::TransitionId::name(transitionId),
                                      timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                                      timingHeader->pulseId());
                    } else {
                        logging::debug("PGPReader saw %s transition @ %u.%09u (%014lx)",
                                       XtcData::TransitionId::name(transitionId),
                                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                                       timingHeader->pulseId());
                    }
                }
                if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
                    logging::critical("%sPGPReader: Jump in complete l1Count %u -> %u | difference %d, tid %s%s",
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

                // To ensure L3 Input Dgrams appear in the batch in sequential
                // order, entry allocation must occur here rather than in the
                // worker threads, the execution order of which may get scrambled
                event->l3InpBuf = tebContributor.allocate(*timingHeader, (void*)((uintptr_t)current));

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
            if (event->l3InpBuf) // else shutting down
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

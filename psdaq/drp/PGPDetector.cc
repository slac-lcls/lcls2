#include <iostream>
#include <fstream>
#include <limits.h>
#include "DataDriver.h"
#include "TimingHeader.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/eb/TebContributor.hh"
#include "DrpBase.hh"
#include "PGPDetector.hh"

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
            Pds::TimingHeader* timingHeader = reinterpret_cast<Pds::TimingHeader*>(pool.dmaBuffers[index]);
            if (pulseId == 0) {
                pulseId = timingHeader->seq.pulseId().value();
            }
            else {
                if (pulseId != timingHeader->seq.pulseId().value()) {
                    printf("Wrong pulse id! expected %lu but got %lu instead\n",
                           pulseId, timingHeader->seq.pulseId().value());
                    return false;
                }
            }
            // check bit 7 in pulseId for error
            bool error = timingHeader->seq.pulseId().control() & (1 << 7);
            if (error) {
                std::cout<<"Error bit in pulseId is set\n";
            }
        }
    }
    return true;
}

void workerFunc(const Parameters& para, MemPool& pool,
                Detector* det,
                SPSCQueue<Batch>& inputQueue, SPSCQueue<Batch>& outputQueue)
{
    Batch batch;
    const unsigned nbuffers = pool.nbuffers();
    uint32_t envMask = 0xffff0000 | uint32_t(para.rogMask);
    while (true) {
        if (!inputQueue.pop(batch)) {
            break;
        }

        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) % nbuffers;
            PGPEvent* event = &pool.pgpEvents[index];
            checkPulseIds(pool, event);

            // make new dgram in the pebble
            XtcData::Dgram* dgram = (XtcData::Dgram*)pool.pebble[index];
            XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
            dgram->xtc.contains = tid;
            dgram->xtc.damage = 0;
            dgram->xtc.extent = sizeof(XtcData::Xtc);

            // get transitionId from the first lane in the event
            int lane = __builtin_ffs(event->mask) - 1;
            uint32_t dmaIndex = event->buffers[lane].index;
            Pds::TimingHeader* timingHeader = (Pds::TimingHeader*)pool.dmaBuffers[dmaIndex];
            XtcData::TransitionId::Value transitionId = timingHeader->seq.service();

            // fill in dgram header
            dgram->seq = timingHeader->seq;
            dgram->env = timingHeader->env & envMask; // Ignore other partitions' RoGs

            // Event
            if (transitionId == XtcData::TransitionId::L1Accept) {
                det->event(*dgram, event);
            }
            // transitions
            else {
                // copy the temporary xtc created on phase 1 of the transition
                // into the real location
                XtcData::Xtc& transitionXtc = det->transitionXtc();
                memcpy(&dgram->xtc, &transitionXtc, transitionXtc.extent);
            }
            // set the src field for the event builders
            dgram->xtc.src = XtcData::Src(det->nodeId);
        }

        outputQueue.push(batch);
    }
}

PGPDetector::PGPDetector(const Parameters& para, MemPool& pool, Detector* det) :
    m_para(para), m_pool(pool), m_terminate(false)
{
    m_nodeId = det->nodeId;
    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    for (int i=0; i<4; i++) {
        if (para.laneMask & (1 << i)) {
            std::cout<<"setting lane  "<<i<<'\n';
            dmaAddMaskBytes(mask, dmaDest(i, 0));
        }
    }
    dmaSetMaskBytes(pool.fd(), mask);

    for (unsigned i=0; i<para.nworkers; i++) {
        m_workerInputQueues.emplace_back(SPSCQueue<Batch>(pool.nbuffers()));
        m_workerOutputQueues.emplace_back(SPSCQueue<Batch>(pool.nbuffers()));
    }


    for (unsigned i = 0; i < para.nworkers; i++) {
        m_workerThreads.emplace_back(workerFunc,
                                   std::ref(para),
                                   std::ref(pool),
                                   det,
                                   std::ref(m_workerInputQueues[i]),
                                   std::ref(m_workerOutputQueues[i]));
    }
}

void PGPDetector::reader(std::shared_ptr<MetricExporter> exporter)
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


    int64_t worker = 0L;
    uint64_t batchId = 0L;
    const unsigned nbuffers = m_pool.nbuffers();
    while (1) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        int32_t ret = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
        for (int b=0; b < ret; b++) {
            int32_t size = dmaRet[b];
            uint32_t index = dmaIndex[b];
            uint32_t lane = (dest[b] >> 8) & 7;
            bytes += size;

            const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
            uint32_t evtCounter = data[5] & 0xffffff;
            uint32_t current = evtCounter % nbuffers;
            PGPEvent* event = &m_pool.pgpEvents[current];

            DmaBuffer* buffer = &event->buffers[lane];
            buffer->size = size;
            buffer->index = index;
            event->mask |= (1 << lane);

            if (event->mask == m_para.laneMask) {
                if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
                    printf("\033[0;31m");
                    printf("Fatal: Jump in complete l1Count %u -> %u | difference %d\n",
                           m_lastComplete, evtCounter, evtCounter - m_lastComplete);
                    printf("data: %08x %08x %08x %08x %08x %08x\n",
                           data[0], data[1], data[2], data[3], data[4], data[5]);
                    printf("\033[0m");
                    throw "Jump in event counter";

                    for (unsigned e=m_lastComplete+1; e<evtCounter; e++) {
                        PGPEvent* brokenEvent = &m_pool.pgpEvents[e % nbuffers];
                        printf("broken event:  %08x\n", brokenEvent->mask);
                        brokenEvent->mask = 0;

                    }
                }
                m_lastComplete = evtCounter;

                nevents++;
                m_batch.size++;

                const Pds::TimingHeader* timingHeader = reinterpret_cast<const Pds::TimingHeader*>(data);
                XtcData::TransitionId::Value transitionId = timingHeader->seq.service();
                uint64_t pid = timingHeader->seq.pulseId().value();

                // send batch to worker if batch is full or if it's a transition
                if (((batchId ^ pid) & ~(m_para.batchSize - 1)) || (transitionId != XtcData::TransitionId::L1Accept)) {
                    m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                    worker++;
                    m_batch.start = evtCounter + 1;
                    m_batch.size = 0;
                    batchId = pid;
                }
            }
        }
    }
}

void PGPDetector::collector(Pds::Eb::TebContributor& tebContributor)
{
    int64_t worker = 0L;
    int64_t counter = 0L;
    Batch batch;
    const unsigned nbuffers = m_pool.nbuffers();
    while (true) {
        if (!m_workerOutputQueues[worker % m_para.nworkers].pop(batch)) {
            break;
        }
        //std::cout<<"collector:  "<<batch.start<<"  "<<batch.size<<'\n';
        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) % nbuffers;
            XtcData::Dgram* dgram = (XtcData::Dgram*)m_pool.pebble[index];
            uint64_t val;
            if (counter % 2 == 0) {
                val = 0xdeadbeef;
            }
            else {
                val = 0xabadcafe;
            }
            // always monitor every event
            val |= 0x1234567800000000ul;
            //val |= ((uint64_t)index) << 32;
            void* buffer = tebContributor.allocate(dgram, (void*)((uintptr_t)index));
            if (buffer) // else this DRP doesn't provide input, or timed out
            {
                MyDgram* dg = new(buffer) MyDgram(*dgram, val, m_nodeId);
                tebContributor.process(dg);
            }
            counter++;
        }
        worker++;
    }
    tebContributor.shutdown();
}

void PGPDetector::resetEventCounter()
{
    m_lastComplete = 0; // 0xffffff;
    m_batch.start = 1;
    m_batch.size = 0;
}

void PGPDetector::shutdown()
{
    m_terminate.store(true, std::memory_order_release);
    std::cout<<"shutting down PGPReader\n";
    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerInputQueues[i].shutdown();
        if (m_workerThreads[i].joinable()) {
            m_workerThreads[i].join();
        }
    }
    std::cout<<"Worker threads finished\n";
    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerOutputQueues[i].shutdown();
    }
}

}

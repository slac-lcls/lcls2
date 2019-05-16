#include <iostream>
#include <fstream>
#include <limits.h>
#include "DataDriver.h"
#include "TimingHeader.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "PGPReader.hh"

namespace Drp {

unsigned nextPowerOf2(unsigned n)
{
    unsigned count = 0;

    if (n && !(n & (n - 1))) {
        return n;
    }

    while( n != 0) {
        n >>= 1;
        count += 1;
    }

    return 1 << count;
}

MemPool::MemPool(const Parameters& para)
{
    fd = open(para.device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<<para.device<<'\n';
        throw "Error opening kcu1500!!\n";
    }

    uint32_t dmaCount, dmaSize;
    dmaBuffers = dmaMapDma(fd, &dmaCount, &dmaSize);
    if (dmaBuffers == NULL ) {
        std::cout<<"Failed to map dma buffers!\n";
        throw "Error calling dmaMapDma!!\n";
    }
    printf("dmaCount %u  dmaSize %u\n", dmaCount, dmaSize);

    // make sure there are more buffers in the pebble than in the pgp driver
    // otherwise the pebble buffers will be overwritten by the pgp event builder
    nbuffers = nextPowerOf2(dmaCount);

    // make the size of the pebble buffer that will contain the datagram equal
    // to the dmaSize times the number of lanes
    unsigned bufferSize = __builtin_popcount(para.laneMask) * dmaSize;
    pebble.resize(nbuffers, bufferSize);
    printf("nbuffer %u  pebble buffer size %u\n", nbuffers, bufferSize);

    pgpEvents.resize(nbuffers);
    for (unsigned i=0; i<para.nworkers; i++) {
        workerInputQueues.emplace_back(SPSCQueue<Batch>(nbuffers));
        workerOutputQueues.emplace_back(SPSCQueue<Batch>(nbuffers));
    }
}

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

void workerFunc(const Parameters&para, MemPool& pool,
                Detector* det,
                SPSCQueue<Batch>& inputQueue, SPSCQueue<Batch>& outputQueue)
{
    Batch batch;
    while (true) {
        if (!inputQueue.pop(batch)) {
            break;
        }

        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) % pool.nbuffers;
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
            dgram->env = timingHeader->env;
            dgram->xtc.src = XtcData::Src(det->nodeId);

            // Event
            if (transitionId == XtcData::TransitionId::L1Accept) {
                det->event(*dgram, event);
            }
            // transitions
            else {
                XtcData::Xtc& transitionXtc = det->transitionXtc();
                memcpy(&dgram->xtc, &transitionXtc, transitionXtc.extent);
            }
        }

        outputQueue.push(batch);
    }
}

PGPReader::PGPReader(const Parameters& para, MemPool& pool,
                     Detector* det) :
    m_para(&para), m_pool(&pool), m_terminate(false)
{
    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    for (int i=0; i<4; i++) {
        if (para.laneMask & (1 << i)) {
            std::cout<<"setting lane  "<<i<<'\n';
            dmaAddMaskBytes(mask, dmaDest(i, 0));
        }
    }
    dmaSetMaskBytes(pool.fd, mask);

    for (unsigned i = 0; i < para.nworkers; i++) {
        m_workerThreads.emplace_back(workerFunc,
                                   std::ref(para),
                                   std::ref(pool),
                                   det,
                                   std::ref(pool.workerInputQueues[i]),
                                   std::ref(pool.workerOutputQueues[i]));
    }
}

void PGPReader::run(std::shared_ptr<MetricExporter> exporter)
{
    // setup monitoring
    uint64_t nevents = 0L;
    uint64_t bytes = 0L;
    std::map<std::string, std::string> labels{{"partition", std::to_string(m_para->partition)}};
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return nevents;});

    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return bytes;});

    exporter->add("drp_port_rcv_rate", labels, MetricType::Rate,
                  [](){return 4*readInfinibandCounter("port_rcv_data");});

    exporter->add("drp_port_xmit_rate", labels, MetricType::Rate,
                  [](){return 4*readInfinibandCounter("port_xmit_data");});

    uint32_t lastComplete = 0; // 0xffffff;
    int64_t worker = 0L;
    Batch batch;
    batch.start = 1;
    batch.size = 0;

    while (1) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        int32_t ret = dmaReadBulkIndex(m_pool->fd, MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
        for (int b=0; b < ret; b++) {
            int32_t size = dmaRet[b];
            uint32_t index = dmaIndex[b];
            uint32_t lane = (dest[b] >> 8) & 7;
            bytes += size;

            const uint32_t* data = (uint32_t*)m_pool->dmaBuffers[index];
            uint32_t evtCounter = data[5] & 0xffffff;
            uint32_t current = evtCounter % m_pool->nbuffers;
            PGPEvent* event = &m_pool->pgpEvents[current];

            DmaBuffer* buffer = &event->buffers[lane];
            buffer->size = size;
            buffer->index = index;
            event->mask |= (1 << lane);

            if (event->mask == m_para->laneMask) {
                if (evtCounter != ((lastComplete + 1) & 0xffffff)) {
                    printf("\033[0;31m");
                    printf("Jump in complete l1Count %u -> %u | difference %d\n",
                           lastComplete, evtCounter, evtCounter - lastComplete);
                    printf("\033[0m");

                    for (unsigned e=lastComplete+1; e<evtCounter; e++) {
                        PGPEvent* brokenEvent = &m_pool->pgpEvents[e % m_pool->nbuffers];
                        // printf("broken event:  %08x\n", brokenEvent->mask);
                        brokenEvent->mask = 0;

                    }
                }
                lastComplete = evtCounter;
                nevents++;
                batch.size++;

                const Pds::TimingHeader* timingHeader = reinterpret_cast<const Pds::TimingHeader*>(data);
                XtcData::TransitionId::Value transitionId = timingHeader->seq.service();

                // send batch to worker if batch is full or if it's a transition
                if ((batch.size == m_para->batchSize) || (transitionId != XtcData::TransitionId::L1Accept)) {
                    m_pool->workerInputQueues[worker % m_para->nworkers].push(batch);
                    worker++;
                    batch.size = 0;
                    batch.start = evtCounter + 1;

                }
            }
        }
    }
}

void PGPReader::shutdown()
{
    m_terminate.store(true, std::memory_order_release);
    std::cout<<"shutting down PGPReader\n";
    for (unsigned i = 0; i < m_para->nworkers; i++) {
        m_pool->workerInputQueues[i].shutdown();
        m_workerThreads[i].join();
    }
    std::cout<<"Worker threads finished\n";
}

}

#include <thread>
#include <cstdio>
#include <chrono>
#include <bitset>
#include "AxisDriver.h"
#include "PGPReader.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Sequence.hh"

using namespace XtcData;

MovingAverage::MovingAverage(int n) : index(0), sum(0), N(n), values(N, 0) {}
int MovingAverage::add_value(int value)
{
    int& oldest = values[index % N];
    sum += value - oldest;
    oldest = value;
    index++;
    return sum;
}

PGPReader::PGPReader(MemPool& pool, int lane_mask, int nworkers) :
    m_pool(pool),
    m_avg_queue_size(nworkers),
    m_pcounter(&m_c1)
{
    std::bitset<32> bs(lane_mask);
    m_nlanes = bs.count();
    m_last_complete = 0;

    m_worker = 0;
    m_nworkers = nworkers;

    m_buffer_mask = m_pool.num_entries - 1;

    // FIXME and make mask from lane_mask
    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    memset(mask, 0xFF, DMA_MASK_SIZE);
    dmaSetMaskBytes(pool.fd, mask);
}

PGPData* PGPReader::process_lane(uint32_t lane, uint32_t index, int32_t size)
{
    Transition* event_header = reinterpret_cast<Transition*>(m_pool.dmaBuffers[index]);
    int j = event_header->evtCounter & m_buffer_mask;
    PGPData* p = &m_pool.pgp_data[j];
    p->buffers[lane].dmaIndex = index;
    p->buffers[lane].size = size;
    p->buffers[lane].data = m_pool.dmaBuffers[index];
    // set bit in lane mask for lane
    p->buffer_mask |= (1 << lane);
    p->counter++;
    if (p->counter == m_nlanes) {
        if (event_header->evtCounter != (m_last_complete + 1)) {
            printf("Jump in complete l1Count %d -> %u\n",
                   m_last_complete, event_header->evtCounter);
            // FIXME clean up broken events and return dma indices
        }
        m_last_complete = event_header->evtCounter;
        return p;
    }
    else {
        return nullptr;
    }
}

void PGPReader::send_to_worker(Pebble* pebble_data)
{
    PebbleQueue* queue;
    // load balanching to find worker to send the event to
    while (true) {
        queue = &m_pool.worker_input_queues[m_worker % m_nworkers];
        int queue_size = queue->guess_size();
        // calculate running mean over the worker queues
        int mean = m_avg_queue_size.add_value(queue_size);
        if (queue_size * m_nworkers - 5 < mean) {
            break;
        }
        m_worker++;
    }
    queue->push(pebble_data);
    m_pool.collector_queue.push(m_worker % m_nworkers);
    m_worker++;
}

void PGPReader::send_all_workers(Pebble* pebble)
{
    for (int i=0; i<m_nworkers; i++) {
        m_pool.worker_input_queues[i].push(pebble);
    }
    // only pass on event from worker 0 to collector
    m_pool.collector_queue.push(0);
}

void PGPReader::run()
{
    Counters* counter = &m_c2;

    int64_t event_count = 0;
    int64_t total_bytes_received = 0;
    while (true) {
        int32_t ret;
        do {
            ret = dmaReadBulkIndex(m_pool.fd, MAX_RET_CNT_C, m_dmaRet, m_dmaIndex, NULL, NULL, m_dmaDest);
        }
        while (ret == 0);

        for (int b=0; b < ret; b++) {
            total_bytes_received += m_dmaRet[b];
            uint32_t dest = m_dmaDest[b] >> 5;
            PGPData* pgp = process_lane(dest, m_dmaIndex[b], m_dmaRet[b]);
            if (pgp) {
                // get first set bit to find index of the first lane
                int index = __builtin_ffs(pgp->buffer_mask) - 1;
                Transition* event_header = reinterpret_cast<Transition*>(pgp->buffers[index].data);
                TransitionId::Value transition_id = event_header->seq.service();
                //printf("Complete evevent:  Transition id %d pulse id %lu  event counter %u\n",
                //        transition_id, event_header->seq.pulseId().value(), event_header->evtCounter);
                Pebble* pebble;
                m_pool.pebble_queue.pop(pebble);
                pebble->pgp_data = pgp;

                send_to_worker(pebble);
                event_count += 1;
                counter->event_count = event_count;
                counter->total_bytes_received = total_bytes_received;
                counter = m_pcounter.exchange(counter, std::memory_order_release);
            }
        }
    }
}

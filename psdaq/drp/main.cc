#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/NamesIter.hh"
#include "psdaq/hdf5/Hdf5Writer.hh"

#include "drp.hh"
#include "spscqueue.hh"
#include "PgpCardMod.h"
#include "Detectors.hh"
#include "EventBuilder.hh"

using PebbleQueue = SPSCQueue<Pebble*>;
using namespace XtcData;
using namespace Pds::Eb;

const int N = 2000000;
const int NWORKERS = 8;

MovingAverage::MovingAverage(int n) : index(0), sum(0), N(n), values(N, 0) {}
int MovingAverage::add_value(int value)
{
    int& oldest = values[index % N];
    sum += value - oldest;
    oldest = value;
    index++;
    return sum;
}

void monitor_pgp(std::atomic<int64_t>& total_bytes_received,
                 std::atomic<int64_t>& event_count)
{
    int64_t old_bytes = total_bytes_received.load(std::memory_order_relaxed);;
    int64_t old_count = event_count.load(std::memory_order_relaxed);
    auto t = std::chrono::steady_clock::now();
    while (1) {
        sleep(1);
        auto oldt = t;
        t = std::chrono::steady_clock::now();

        int64_t new_bytes = total_bytes_received.load(std::memory_order_relaxed);
        if (new_bytes == -1) {
            break;
        }
        int64_t new_count = event_count.load(std::memory_order_relaxed);

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double data_rate = double(new_bytes - old_bytes) / duration;
        double event_rate = double(new_count - old_count) / duration * 1.0e3;
        printf("Event rate %.2f kHz    Data rate  %.2f MB/s\n", event_rate, data_rate);
        old_bytes = new_bytes;
        old_count = new_count;
    }
}

// send pgp data to worker thread and does load balancing to even work load across worker threads
class WorkerSender
{
public:
    WorkerSender(int nworkers, std::vector<PebbleQueue>* worker_input_queues, SPSCQueue<int>* collector_queue) :
       _avg_queue_size(nworkers)
    {
        _nworkers = nworkers;
        _worker = 0;
        _worker_input_queues = worker_input_queues;
        _collector_queue = collector_queue;
    }

    void send_to_worker(Pebble* pebble_data)
    {
        PebbleQueue* queue;
        while (true) {
            queue = &(*_worker_input_queues)[_worker % _nworkers];
            int queue_size = queue->guess_size();
            // calculate running mean over the worker queues
            int mean = _avg_queue_size.add_value(queue_size);
            if (queue_size * _nworkers - 5 < mean) {
                break;
            }
            _worker++;
        }
        queue->push(pebble_data);
        _collector_queue->push(_worker % _nworkers);
        _worker++;
    }
private:
    int _nworkers;
    MovingAverage _avg_queue_size;
    uint64_t _worker;
    std::vector<PebbleQueue>* _worker_input_queues;
    SPSCQueue<int>* _collector_queue;
};

class PGPEventBuilder
{
public:
    PGPEventBuilder(int queue_size, int nlanes) : _pgp_data(queue_size), _nlanes(nlanes)
    {
        assert((queue_size & (queue_size - 1)) == 0);
        _buffer_mask = queue_size - 1;
        _last_complete = -1;
    }
    PGPData* process_lane(Transition* event_header, int lane, int index, int size)
    {
        // event builder
        int j = event_header->evtCounter & _buffer_mask;
        PGPData* p = &_pgp_data[j];
        PGPBuffer* buffer = &p->buffers[lane];
        buffer->dma_index = index;
        buffer->length = size;

        // set bit in lane mask for lane
        p->lane_mask |= (1 << lane);
        p->counter++;

        if (p->counter == _nlanes) {
            if (event_header->evtCounter != (_last_complete + 1)) {
                printf("Jump in complete evtCounter %d -> %u\n",
                        _last_complete, event_header->evtCounter);
                // FIXME clean up broken events and return dma indices
            }
            _last_complete = event_header->evtCounter;
            return p;
        }
        else {
            return nullptr;
        }
    }
 private:
    std::vector<PGPData> _pgp_data;
    int _nlanes;
    int _buffer_mask;
    unsigned _last_complete;
};

void pgp_reader(SPSCQueue<uint32_t>& index_queue, PebbleQueue& pgp_queue, uint32_t** dma_buffers, SPSCQueue<int>& collector_queue, std::vector<PebbleQueue>& worker_input_queues, std::vector<unsigned> lanes)
{
    int nlanes = lanes.size();

    // open pgp card fd
    char dev_name[128];
    snprintf(dev_name, 128, "/dev/pgpcardG3_0_%u", lanes[0]+1);
    int fd = open(dev_name, O_RDWR);
    if (fd < 0) {
        std::cout << "Failed to open pgpcard" << dev_name << std::endl;
    }

    // setup receiving from multiple channels
    PgpCardTx pgpCardTx;
    pgpCardTx.model = sizeof(&pgpCardTx);
    pgpCardTx.size = sizeof(PgpCardTx);
    pgpCardTx.cmd = IOCTL_Add_More_Ports;
    pgpCardTx.data = reinterpret_cast<uint32_t*>(nlanes-1);
    write(fd, &pgpCardTx, sizeof(PgpCardTx));

    WorkerSender worker_sender(worker_input_queues.size(), &worker_input_queues, &collector_queue);
    PGPEventBuilder pgp_builder(8192, nlanes);

    std::atomic<int64_t> total_bytes_received(0L);
    std::atomic<int64_t> event_count(0L);
    std::thread monitor_thread(monitor_pgp, std::ref(total_bytes_received), std::ref(event_count));

    PgpCardRx pgp_card;
    pgp_card.model = sizeof(&pgp_card);
    pgp_card.maxSize = 100000;

    int i = 0;
    uint64_t bytes_received = 0UL;
    while (i < N) {
        uint32_t index;
        if (!index_queue.pop(index)) {
            std::cout<<"Error in getting new index\n";
            return;
        }

        // read from pgp card
        pgp_card.data = dma_buffers[index];
        unsigned int size = read(fd, &pgp_card, sizeof(pgp_card));
        if (size <= 0) {
             std::cout << "Error in reading from pgp card!" << std::endl;
        } else if (size == pgp_card.maxSize) {
            std::cout << "Warning! Package size bigger than the maximum size!" << std::endl;
        }
        bytes_received += size*4;

        int lane = pgp_card.pgpLane;
        Transition* event_header = reinterpret_cast<Transition*>(dma_buffers[index]);

        //printf("pulse id: %lu  lane: %d  evtCounter: %d\n", event_header->seq.pulseId().value(), lane, event_header->evtCounter);

        PGPData* pgp_data = pgp_builder.process_lane(event_header, lane, index, size);
        if (pgp_data) {
            i++;
            Pebble* pebble;
            pgp_queue.pop(pebble);
            pebble->pgp_data = pgp_data;
            worker_sender.send_to_worker(pebble);

            // update pgp metrics
            uint64_t temp = event_count.load(std::memory_order_relaxed) + 1;
            event_count.store(temp, std::memory_order_relaxed);
            temp = total_bytes_received.load(std::memory_order_relaxed) + bytes_received;
            total_bytes_received.store(temp, std::memory_order_relaxed);
            bytes_received = 0UL;
        }
    }


    // shutdown monitor thread
    total_bytes_received.store(-1, std::memory_order_relaxed);
    monitor_thread.join();
    close(fd);
}

bool check_pulseIds(PGPData* pgp_data, uint32_t** dma_buffers)
{
    uint64_t pulse_id = 0;
    for (int l=0; l<8; l++) {
        if (pgp_data->lane_mask  & (1 << l)) {
            uint32_t index = pgp_data->buffers[l].dma_index;
            Transition* event_header = reinterpret_cast<Transition*>(dma_buffers[index]);
            if (pulse_id == 0) {
                pulse_id = event_header->seq.pulseId().value();
            }
            else {
                if (pulse_id != event_header->seq.pulseId().value()) {
                    printf("Wrong pulse id! expected %lu but got %lu instead\n", pulse_id, event_header->seq.pulseId().value());
                    return false;
                }
            }
        }
    }
    return true;
}

void worker(PebbleQueue& worker_input_queue, PebbleQueue& worker_output_queue, uint32_t** dma_buffers, int rank, std::vector<unsigned> lanes)
{
    std::vector<NameIndex> namesVec;

    int64_t counter = 0;
    while (true) {
        Pebble* pebble_data;
        if (!worker_input_queue.pop(pebble_data)) {
            break;
        }

        if (!pebble_data->pgp_data->damaged) {
            check_pulseIds(pebble_data->pgp_data, dma_buffers);

            Dgram& dgram = *(Dgram*)pebble_data->fex_data();
            TypeId tid(TypeId::Parent, 0);
            dgram.xtc.contains = tid;
            dgram.xtc.damage = 0;
            dgram.xtc.extent = sizeof(Xtc);

            // Do actual work here
            // configure transition
            if (counter == 0) {
                add_roi_names(dgram.xtc, namesVec);
                // add_hsd_Snames(dgram.xtc, namesVec);
            }
            // making real fex data for event
            else {
                // need to make more robust: have to match this index
                // to pick up the correct array element in add_NNN_names
                unsigned nameId = 0;
                roiExample(dgram.xtc, namesVec, nameId, pebble_data, dma_buffers);
                // hsdExample(dgram.xtc, namesVec, nameId, pebble_data, dma_buffers, lanes);
            }
        }

        worker_output_queue.push(pebble_data);
        counter++;
    }
    printf("Thread %d processed %lu events\n", rank, counter);
}

void pin_thread(const pthread_t& th, int cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    int rc = pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cout << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}


class XtcFile
{
public:
    XtcFile(const char* fname)
    {
        file = fopen(fname, "w");
        if (!file) {
            printf("Error opening file %s\n",fname);
            _exit(1);
        }
    }
    ~XtcFile()
    {
        fclose(file);
    }
    int save(Dgram& dgram)
    {
        if (fwrite(&dgram, sizeof(Dgram) + dgram.xtc.sizeofPayload(), 1, file) != 1) {
            printf("Error writing to output xtc file. %p %lu\n",&dgram, sizeof(Dgram) + dgram.xtc.sizeofPayload());
            perror("save");
            _exit(1);
        }
        fflush(file);
        return 0;
    }
    int saveIov(Dgram& dgram, const struct iovec* iov, int iovcnt)
    {
        unsigned iovlen=0;
        for (int i=0; i<iovcnt; i++) {
            iovlen+=iov[i].iov_len;
        }
        if (fwrite(&dgram, sizeof(Dgram) + dgram.xtc.sizeofPayload()-iovlen, 1, file) != 1) {
            printf("Error writing to output xtc file. %p %lu\n",&dgram,sizeof(Dgram) + dgram.xtc.sizeofPayload()-iovlen);
            perror("saveIov hdr");
            _exit(1);
        }
        for (int i=0; i<iovcnt; i++) {
            if (iov[i].iov_len==0) {
                printf("zero len iov\n");
                continue;
            }
            if (fwrite(iov[i].iov_base, iov[i].iov_len, 1, file) != 1) {
                printf("Error writing IOV to output xtc file %p %lu.\n",iov[i].iov_base, iov[i].iov_len);
                perror("saveIov");
                _exit(1);
            }
        }
        fflush(file);
        return 0;
    }
private:
    FILE* file;
};

int main()
{
    int queue_size = 8192;
    std::vector<unsigned> lanes = {4, 5, 6, 7}; // must be contiguous

    // Pds::StringList peers;
    // peers.push_back("172.21.52.136"); //acc05
    // Pds::StringList ports;
    // ports.push_back("32768");
    // EbLfClient myEbLfClient(peers,ports);

    // MyBatchManager myBatchMan(myEbLfClient);

    // unsigned timeout = 120;
    // myEbLfClient.connect(ContribId, timeout, myBatchMan.batchRegion(), myBatchMan.batchRegionSize());

    // printf("*** myEb %p %zd\n",myBatchMan.batchRegion(), myBatchMan.batchRegionSize());
    // // start eb receiver thread
    // std::thread eb_rcvr_thread(eb_rcvr, std::ref(myBatchMan));

    // pin main thread
    pin_thread(pthread_self(), 1);

    // index_queue goes away with the new pgp driver in the dma streaming mode
    SPSCQueue<uint32_t> index_queue(queue_size);
    SPSCQueue<Pebble*> pebble_queue(queue_size);
    uint32_t** dma_buffers;
    std::vector<Pebble> pebble(queue_size);

    std::vector<PebbleQueue> worker_input_queues;
    std::vector<PebbleQueue> worker_output_queues;
    for (int i = 0; i < NWORKERS; i++) {
        worker_input_queues.emplace_back(PebbleQueue(queue_size));
        worker_output_queues.emplace_back(PebbleQueue(queue_size));
    }

    SPSCQueue<int> collector_queue(queue_size);

    // buffer size in elements of 4 byte units
    int64_t buffer_element_size = 100000;
    int64_t buffer_size = queue_size * buffer_element_size;
    std::cout << "buffer size:  " << buffer_size * 4 / 1.e9 << " GB" << std::endl;
    dma_buffers  = new uint32_t*[queue_size];
    for (int i = 0; i < queue_size; i++) {
        index_queue.push(i);
        pebble_queue.push(&pebble[i]);
        dma_buffers[i] = new uint32_t[buffer_element_size];
    }

    // start pgp reader thread
    std::thread pgp_thread(pgp_reader, std::ref(index_queue), std::ref(pebble_queue), dma_buffers, std::ref(collector_queue),
                           std::ref(worker_input_queues), lanes);
    pin_thread(pgp_thread.native_handle(), 2);

    // start worker threads
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < NWORKERS; i++) {
        worker_threads.emplace_back(worker, std::ref(worker_input_queues[i]),
                                    std::ref(worker_output_queues[i]), dma_buffers, i, lanes);
        pin_thread(worker_threads[i].native_handle(), 3 + i);
    }

    // XtcFile xtcfile("/reg/neh/home/cpo/data.xtc");
    // NamesIter namesiter;
    // HDF5File h5file("/u1/cpo/data.h5", namesiter.namesVec());

    // start loop for the collector to collect results from the workers in the same order the events arrived over pgp
    iovec iov[8];
    for (int i = 0; i < N; i++) {
        int worker;
        collector_queue.pop(worker);

        Pebble* pebble_data;
        worker_output_queues[worker].pop(pebble_data);

        Dgram& dgram = *reinterpret_cast<Dgram*>(pebble_data->fex_data());

        // uint64_t val;
        // if (i%3==0) {
        //     val = 0xdeadbeef;
        // } else {
        //     val = 0xabadcafe;
        // }
        // MyDgram dg(i,val);
        // myBatchMan.process(&dg);

        // if (i==0) {
        //     xtcfile.save(dgram);
        // } else {
        //     PGPBuffer* buffers = pebble_data->pgp_data->buffers;
        //     for (unsigned ilane=0; ilane<lanes.size(); ilane++) {
        //         iov[ilane].iov_len=buffers[lanes[ilane]].length*sizeof(uint32_t);
        //         uint32_t dma_index = buffers[lanes[ilane]].dma_index;
        //         iov[ilane].iov_base = reinterpret_cast<void*>(dma_buffers[dma_index]);
        //     }
        //     xtcfile.saveIov(dgram, iov, lanes.size());
        // }

        // if (i == 0) {
        //     namesiter.iterate(&dgram.xtc);
        // }
        // else {
        //     h5file.save(dgram);
        // }

        // return dma indices to dma buffer pool
        for (int l=0; l<8; l++) {
            if (pebble_data->pgp_data->lane_mask  & (1 << l)) {
                index_queue.push(pebble_data->pgp_data->buffers[l].dma_index);
            }
        }
        pebble_data->pgp_data->counter = 0;
        pebble_data->pgp_data->lane_mask = 0;
        pebble_queue.push(pebble_data);
    }

    // shutdown worker queues and wait for threads to finish
    for (int i = 0; i < NWORKERS; i++) {
        worker_input_queues[i].shutdown();
        worker_threads[i].join();
    }
    for (int i = 0; i < NWORKERS; i++) {
        worker_output_queues[i].shutdown();
    }

    // buffer_queue.shutdown();
    pgp_thread.join();
    for (int i=0; i<queue_size; i++) {
        delete [] dma_buffers[i];
    }
    delete [] dma_buffers;
    return 0;
}

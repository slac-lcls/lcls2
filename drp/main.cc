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

#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EbFtClient.hh"
#include "psdaq/eb/EbFtServer.hh"

#include "main.hh"
#include "spscqueue.hh"
#include "PgpCardMod.h"

using PebbleQueue = SPSCQueue<Pebble*>;
using namespace XtcData;
using namespace Pds::Eb;

// these parameters must agree with the server side
unsigned maxBatches = 1000; // size of the pool of batches
unsigned maxEntries = 10; // maximum number of events in a batch
unsigned BatchSizeInPulseIds = 8; // age of the batch. should never exceed maxEntries above, must be a power of 2

unsigned EbId = 0; // from 0-63, maximum number of event builders
unsigned ContribId = 0; // who we are

class TheSrc : public Src
{
public:
    TheSrc(Level::Type level, unsigned id) :
        Src(level)
    {
        _log |= id;
    }
};

class MyDgram : public Dgram {
public:
    MyDgram(unsigned pulseId, uint64_t val) {
        seq = Sequence(Sequence::Event, TransitionId::L1Accept, ClockTime(), TimeStamp(pulseId));
        env = Env(0);
        xtc = Xtc(TypeId(TypeId::Data, 0), TheSrc(Level::Segment, ContribId));
        _data = val;
        xtc.alloc(sizeof(_data));
    }
private:
    uint64_t _data;
};

size_t maxSize = sizeof(MyDgram);

class MyBatchManager: public BatchManager {
public:
    MyBatchManager(EbFtClient& ebFtClient) :
        BatchManager(BatchSizeInPulseIds, maxBatches, maxEntries, maxSize),
        _ebFtClient(ebFtClient)
    {}
    void post(Batch* batch) {
        _ebFtClient.post(batch->buffer(), batch->extent(), EbId, batch->index() * maxBatchSize());
    }
private:
    EbFtClient& _ebFtClient;
};

struct EventHeader {
    uint64_t pulseId;
    uint64_t timeStamp;
    uint32_t trigTag;
    uint32_t l1Count;
    unsigned rawSamples:24;
    unsigned channelMask:8;
    uint32_t reserved;
};

const int N = 2000000;
const int NWORKERS = 1;

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
                      std::atomic<int64_t>& event_count, std::atomic<int64_t>& complete_count)
{
    int64_t old_bytes = total_bytes_received.load(std::memory_order_relaxed);;
    int64_t old_count = event_count.load(std::memory_order_relaxed);
    int64_t old_complete = complete_count.load(std::memory_order_release);
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
        int64_t new_complete = complete_count.load(std::memory_order_release);

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double data_rate = double(new_bytes - old_bytes) / duration;
        double event_rate = double(new_count - old_count) / duration * 1.0e3;
        int denominator = (new_count - old_count);
        double complete_ratio = 0.0;
        if (denominator > 0) {
          complete_ratio  = double(new_complete - old_complete) / double(new_count - old_count);
        }
        printf("Event rate %.2f kHz    Data rate  %.2f MB/s    Complete event ratio %.2f%% \n", event_rate, data_rate, complete_ratio);
        old_bytes = new_bytes;
        old_count = new_count;
        old_complete = new_complete;

    }
}

// send pgp data to worker thread and does load balancing to even work load across worker threads
class WorkerSender
{
public:
    WorkerSender(int nworkers) :_nworkers(nworkers), _avg_queue_size(nworkers), _worker(0) {}
    void send_to_worker(std::vector<PebbleQueue>& worker_input_queues, SPSCQueue<int>& collector_queue, Pebble* pebble_data)
    {
    PebbleQueue* queue;
    while (true) {
        queue = &worker_input_queues[_worker % _nworkers];
        int queue_size = queue->guess_size();
        // calculate running mean over the worker queues
        int mean = _avg_queue_size.add_value(queue_size);
        if (queue_size * _nworkers - 5 < mean) {
            break;
        }
        _worker++;
    }
    queue->push(pebble_data);
    collector_queue.push(_worker % _nworkers);
    _worker++;
}
private:
    int _nworkers;
    MovingAverage _avg_queue_size;
    uint64_t _worker;
};


void eb_rcvr(MyBatchManager& myBatchMan)
{
    std::string srvPort = "32832"; // add 64 to the client base
    unsigned numEb = 1;
    size_t maxBatchSize = sizeof(Dgram) + maxEntries * maxSize;
    EbFtServer myEbFtServer(srvPort, numEb, maxBatches * maxBatchSize, EbFtServer::PEERS_SHARE_BUFFERS);
    printf("*** rcvr %d %zd\n",maxBatches,maxBatchSize);
    myEbFtServer.connect(ContribId);
    unsigned nreceive = 0;
    unsigned none = 0;
    unsigned nzero = 0;
    while(1) {
        uint64_t data;
        if (myEbFtServer.pend(&data))  continue;
        const Dgram* batch = (const Dgram*)data;
        unsigned idx = ((const char*)batch - myEbFtServer.base()) / maxBatchSize;
        // printf("received batch %p %d\n",batch,idx);
        const Batch*  input  = myBatchMan.batch(idx);

        const Dgram*  result = (const Dgram*)batch->xtc.payload();
        const Dgram*  last   = (const Dgram*)batch->xtc.next();
        while(result != last) {
            nreceive++;
            // printf("--- result %lx\n",*(uint64_t*)(result->xtc.payload()));
            uint64_t val = *(uint64_t*)(result->xtc.payload());
            result = (Dgram*)result->xtc.next();
            if (val==0) {
                nzero++;
            } else if (val==1) {
                none++;
            } else {
                printf("error %ld\n",val);
            }
            if (nreceive%10000==0) printf("%d %d %d\n",nreceive,none,nzero);
        }
        delete input;
    }
}

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

    WorkerSender worker_sender(worker_input_queues.size());

    std::atomic<int64_t> total_bytes_received(0L);
    std::atomic<int64_t> event_count(0L);
    std::atomic<int64_t> complete_count(0UL);
    std::thread monitor_thread(monitor_pgp, std::ref(total_bytes_received), std::ref(event_count), std::ref(complete_count));

    PgpCardRx pgp_card;
    pgp_card.model = sizeof(&pgp_card);
    pgp_card.maxSize = 100000;
   
     // event builder data structures
    int64_t last_event[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    Pebble* local_buffer[1024];
    int64_t current_event = -1;

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
        unsigned int ret = read(fd, &pgp_card, sizeof(pgp_card));
        if (ret <= 0) {
             std::cout << "Error in reading from pgp card!" << std::endl;
        } else if (ret == pgp_card.maxSize) {
            std::cout << "Warning! Package size bigger than the maximum size!" << std::endl;
        }
        bytes_received += ret*4;
        
        
        // event build pgp lanes
        int lane = pgp_card.pgpLane;
        EventHeader* event_header = reinterpret_cast<EventHeader*>(dma_buffers[index]);
        // std::cout<<"pulse id:  "<<event_header->pulseId<<"  "<<lane<<"  "<<event_header->l1Count<<std::endl;

        // create new event
        if (event_header->l1Count > current_event) {
            // std::cout<<"Create new pgp data\n";
            current_event = event_header->l1Count;
            Pebble** pebble_data = &local_buffer[current_event & 1023];
            pgp_queue.pop(*pebble_data);
            PGPData* pgp = (*pebble_data)->pgp_data();
            // FIXME return old dma indices
            pgp->nlanes = 0;
            pgp->lane_count = 0;
            pgp->pulse_id = event_header->pulseId;
        }

        // non consectutive event_count, missing event contribution
        if ((event_header->l1Count != (last_event[lane] + 1) && (last_event[lane] > 0))) {
            // this event is missing contribution
            // FIXME handle jumping over more than one event
            int64_t index = last_event[lane] + 1;
            Pebble* p = local_buffer[index & 1023];
            PGPData* incomplete_event = p->pgp_data();
            incomplete_event->lane_count++;
            if (incomplete_event->lane_count == nlanes) {
                worker_sender.send_to_worker(worker_input_queues, collector_queue, p);
                i++;
                uint64_t temp = event_count.load(std::memory_order_relaxed) + 1;
                event_count.store(temp, std::memory_order_relaxed);
            }
            std::cout<<"Non consecutive:  "<<event_header->l1Count<<"  "<<last_event[lane]<<std::endl;
        }


        last_event[lane] = event_header->l1Count;
        Pebble* p = local_buffer[last_event[lane] & 1023];
        PGPData* pgp_data = p->pgp_data();
        
        // update pgp data
        PGPBuffer* buffer = &pgp_data->buffers[lane];
        buffer->dma_index = index;
        buffer->length = ret;
        pgp_data->nlanes++;
        pgp_data->lane_count++;

        // received all buffers and event is complete
        if (pgp_data->lane_count == nlanes) {
            // std::cout<<"Complete event\n";
            worker_sender.send_to_worker(worker_input_queues, collector_queue, p);
            i++;

             // update pgp metrics
            uint64_t temp = event_count.load(std::memory_order_relaxed) + 1;
            event_count.store(temp, std::memory_order_relaxed);
            temp = total_bytes_received.load(std::memory_order_relaxed) + bytes_received;
            total_bytes_received.store(temp, std::memory_order_relaxed);
            temp = complete_count.load(std::memory_order_release) + 1;
            complete_count.store(temp, std::memory_order_release);
            bytes_received = 0UL;
         }
    }


    // shutdown monitor thread
    total_bytes_received.store(-1, std::memory_order_relaxed);
    monitor_thread.join();

    close(fd);
}

void hsdExample(Xtc& parent, NameIndex& nameindex, unsigned nameId, Pebble* pebble_data, uint32_t** dma_buffers, std::vector<unsigned>& lanes)
{
    char chan_name[8];
    CreateData hsd(parent, nameindex, nameId);
    PGPBuffer* buffers = pebble_data->pgp_data()->buffers;
    uint32_t shape[1];
    for (unsigned i=0; i<lanes.size(); i++) {
        sprintf(chan_name,"chan%d",i);
        shape[0] = buffers[lanes[i]].length*sizeof(uint32_t);
        hsd.set_array_shape(chan_name, shape);
    }
}

void roiExample(Xtc& parent, NameIndex& nameindex, unsigned nameId, Pebble* pebble_data, uint32_t** dma_buffers)
{
    CreateData fex(parent, nameindex, nameId);

    uint16_t* ptr = (uint16_t*)fex.get_ptr();
    unsigned shape[Name::MaxRank];
    shape[0] = 30;
    shape[1] = 30;
    uint32_t dma_index = pebble_data->pgp_data()->buffers[0].dma_index;
    uint16_t* img = reinterpret_cast<uint16_t*>(dma_buffers[dma_index]);
    for (unsigned i=0; i<shape[0]*shape[1]; i++) {
        ptr[i] = img[i];
    }
    fex.set_array_shape("array_fex",shape);
}

void add_hsd_names(Xtc& parent, std::vector<NameIndex>& namesVec) {
    Alg alg("hsd",1,2,3);
    Names& fexNames = *new(parent) Names(alg,"raw","hsd1");

    fexNames.add("chan0", Name::UINT8, parent, 1);
    fexNames.add("chan1", Name::UINT8, parent, 1);
    fexNames.add("chan2", Name::UINT8, parent, 1);
    fexNames.add("chan3", Name::UINT8, parent, 1);
    namesVec.push_back(NameIndex(fexNames));
}

void add_roi_names(Xtc& parent, std::vector<NameIndex>& namesVec) {
    Alg alg("roi",1,0,0);
    Names& fexNames = *new(parent) Names(alg,"fex","cspad");

    fexNames.add("array_fex", Name::UINT16, parent, 2);
    namesVec.push_back(NameIndex(fexNames));
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

        Dgram& dgram = *(Dgram*)pebble_data->fex_data();
        TypeId tid(TypeId::Parent, 0);
        dgram.xtc.contains = tid;
        dgram.xtc.damage = 0;
        dgram.xtc.extent = sizeof(Xtc);

        // check pulseId
        uint64_t pulse_id;
        PGPData* pgp_data = pebble_data->pgp_data();
        if (pgp_data->nlanes == 3) {
            uint32_t index = pgp_data->buffers[4].dma_index;
            EventHeader* event_header = reinterpret_cast<EventHeader*>(dma_buffers[index]);
            pulse_id = event_header->pulseId;
            for (int i = 1; i<3; i++) {
                index = pgp_data->buffers[lanes[i]].dma_index;
                event_header = reinterpret_cast<EventHeader*>(dma_buffers[index]);
                if (pulse_id != event_header->pulseId) {
                    std::cout<<"Wrong pulse id\n";
                }
            }
        }

        // Do actual work here
        // configure transition
        if (counter == 0) {
            // add_roi_names(dgram.xtc, namesVec);
            add_hsd_names(dgram.xtc, namesVec);
        }
        // making real fex data for event
        else {
            // need to make more robust: have to match this index
            // to pick up the correct array element in add_NNN_names
            unsigned nameId = 0;
            // roiExample(dgram.xtc, namesVec[nameId], nameId, pebble_data, dma_buffers);
            hsdExample(dgram.xtc, namesVec[nameId], nameId, pebble_data, dma_buffers, lanes);
        }

        worker_output_queue.push(pebble_data);
        counter++;
    }
    std::cout << "Thread " << rank << " processed " << counter << " events" << std::endl;
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
    int queue_size = 16384;
    std::vector<unsigned> lanes = {0,1,2,3}; // must be contiguous

    // StringList peers;
    // peers.push_back("172.21.52.136"); //acc05
    // StringList port;
    // port.push_back("32768");
    // size_t rmtSize = maxBatches * (sizeof(Dgram) + maxEntries * maxSize);
    // printf("*** rmtsize %zd\n",rmtSize);
    // EbFtClient myEbFtClient(peers,port,rmtSize);

    // MyBatchManager myBatchMan(myEbFtClient);

    // unsigned timeout = 120;
    // myEbFtClient.connect(ContribId, timeout);

    // myEbFtClient.registerMemory(myBatchMan.batchRegion(), myBatchMan.batchRegionSize());
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

    XtcFile xtcfile("/drpffb/cpo/data.xtc");
    NamesIter namesiter;
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

        if (i==0) {
            xtcfile.save(dgram);
        } else {
            PGPBuffer* buffers = pebble_data->pgp_data()->buffers;
            for (unsigned ilane=0; ilane<lanes.size(); ilane++) {
                iov[ilane].iov_len=buffers[lanes[ilane]].length*sizeof(uint32_t);
                uint32_t dma_index = buffers[lanes[ilane]].dma_index;
                iov[ilane].iov_base = reinterpret_cast<void*>(dma_buffers[dma_index]);
            }
            xtcfile.saveIov(dgram, iov, lanes.size());
        }
        // if (i == 0) {
        //     namesiter.iterate(&dgram.xtc);
        // }
        // else {
        //     h5file.save(dgram);
        // }

        // return dma indices to dma buffer pool
        for (unsigned int b=0; b<pebble_data->pgp_data()->nlanes; b++) {
            index_queue.push(pebble_data->pgp_data()->buffers[b].dma_index);
        }
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

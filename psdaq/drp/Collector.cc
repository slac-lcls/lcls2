#include <linux/limits.h>
#include <rdma/fi_domain.h>
#include <thread>
#include <cassert>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <zmq.h>
#include "Collector.hh"
using namespace XtcData;
using namespace Pds::Eb;

// these parameters must agree with the server side
unsigned maxBatches = 8192; // size of the pool of batches
unsigned maxEntries = 64 ; // maximum number of events in a batch
unsigned BatchSizeInPulseIds = 64; // age of the batch. should never exceed maxEntries above, must be a power of 2
unsigned EbId = 0; // from 0-63, maximum number of event builders
size_t maxSize = sizeof(MyDgram);

MyBatchManager::MyBatchManager(Pds::Eb::EbLfClient& ebFtClient, unsigned contributor_id) :
    Pds::Eb::BatchManager(BatchSizeInPulseIds, maxBatches, maxEntries, maxSize),
    inflight_count(0),
    _ebLfClient(ebFtClient),
    _contributor_id(contributor_id)
{}

void MyBatchManager::post(const Pds::Eb::Batch* batch)
{
    _ebLfClient.post(EbId, batch->datagram(), batch->extent(),
                     batch->index() * maxBatchSize(),
                     (_contributor_id << 24) + batch->index());
    inflight_count.fetch_add(1);
}

MyDgram::MyDgram(Sequence& sequence, uint64_t val, unsigned contributor_id)
{
    seq = sequence;
    xtc = Xtc(TypeId(TypeId::Data, 0), TheSrc(Level::Segment, contributor_id));
    _data = val;
    xtc.alloc(sizeof(_data));
}

static size_t calcBatchSize(unsigned maxEntries, size_t maxSize)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  size_t size      = sizeof(Dgram) + maxEntries * maxSize;
  size             = alignment * ((size + alignment - 1) / alignment);
  return size;
}

static void* allocBatchRegion(unsigned maxBatches, size_t maxBatchSize)
{
  size_t   alignment = sysconf(_SC_PAGESIZE);
  size_t   size      = maxBatches * maxBatchSize;
  assert((size & (alignment - 1)) == 0);
  void*    region    = nullptr;
  int      ret       = posix_memalign(&region, alignment, size);
  if (ret)
  {
    perror("posix_memalign");
    return nullptr;
  }

  return region;
}

// collects events from the workers and sends them to the event builder
void collector(MemPool& pool, Parameters& para, MyBatchManager& myBatchMan)
{
    void* context = zmq_ctx_new();
    void* socket = zmq_socket(context, ZMQ_PUSH);
    zmq_connect(socket, "tcp://localhost:5559");

    printf("*** myEb %p %zd\n",myBatchMan.batchRegion(), myBatchMan.batchRegionSize());
    // // start eb receiver thread
    std::thread eb_rcvr_thread(eb_receiver, std::ref(myBatchMan), std::ref(pool),
                               std::ref(para));

    int i = 0;
    while (true) {
        int worker;
        pool.collector_queue.pop(worker);
        Pebble* pebble;
        pool.worker_output_queues[worker].pop(pebble);

        int index = __builtin_ffs(pebble->pgp_data->buffer_mask) - 1;
        Transition* event_header = reinterpret_cast<Transition*>(pebble->pgp_data->buffers[index]->virt);
        TransitionId::Value transition_id = event_header->seq.service();
         if (transition_id == 2) {
            printf("Collector saw configure transition\n");
        }
        // pass non L1 accepts to control level
        if (transition_id != 0) {
            Dgram* dgram = (Dgram*)pebble->fex_data();
            zmq_send(socket, dgram, sizeof(Dgram) + dgram->xtc.sizeofPayload(), 0);
            printf("Send transition over zeromq socket\n");
        }

        // printf("Collector:  Transition id %d pulse id %lu event counter %u \n",
        //        transition_id, event_header->seq.pulseId().value(), event_header->evtCounter);


        Dgram& dgram = *reinterpret_cast<Dgram*>(pebble->fex_data());
        uint64_t val;
        if (i%5 == 0) {
            val = 0xdeadbeef;
        } else {
            val = 0xabadcafe;
        }
        MyDgram dg(dgram.seq, val, para.contributor_id);
        /* 
        Dgram dg(dgram);
        dg.xtc.src = TheSrc(Level::Segment, para.contributor_id);
        dg.xtc.extent = sizeof(dg.xtc);
        uint64_t* payload = (uint64_t*)dg.xtc.alloc(8);
        *payload = val;
        */
        pool.output_queue.push(pebble);
        myBatchMan.process(&dg);
        i++;
    }
}

void eb_receiver(MyBatchManager& myBatchMan, MemPool& pool, Parameters& para)
{
    char* ifAddr = nullptr;
    unsigned port = 32832 + para.contributor_id;
    std::string srvPort = std::to_string(port);
    unsigned numEb = 1;
    size_t maxBatchSize = calcBatchSize(maxEntries, maxSize);
    void* region = allocBatchRegion(maxBatches, maxBatchSize);
    EbLfServer myEbLfServer(ifAddr, srvPort, numEb);
    printf("*** rcvr %d %zd\n", maxBatches, maxBatchSize);
    myEbLfServer.connect(para.contributor_id, region, maxBatches * maxBatchSize,
                         EbLfServer::PEERS_SHARE_BUFFERS);
    unsigned nreceive = 0;

    char file_name[PATH_MAX];
    snprintf(file_name, PATH_MAX, "%s/data-%02d.xtc", para.output_dir.c_str(), para.contributor_id);
    FILE* xtcFile = fopen(file_name, "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
        return;
    }

    while(1) {
        fi_cq_data_entry wc;
        if (myEbLfServer.pend(&wc))  continue;
        unsigned     idx   = wc.data & 0x00ffffff;
        unsigned     srcId = wc.data >> 24;
        const Dgram* batch = (const Dgram*)(myEbLfServer.lclAdx(srcId, idx * maxBatchSize));

        myEbLfServer.postCompRecv(srcId);

        // printf("received batch %p %d\n",batch,idx);
        const Batch* input  = myBatchMan.batch(idx);
        const Dgram* result = (const Dgram*)batch->xtc.payload();
        const Dgram* last   = (const Dgram*)batch->xtc.next();
        while(result != last) {
            nreceive++;
            uint64_t eb_decision = *(uint64_t*)(result->xtc.payload());
            // printf("eb decision %lu\n", eb_decision);
            Pebble* pebble;
            if (!pool.output_queue.pop(pebble)) {
                printf("output_queue empty and finished\n");
            }

            int index = __builtin_ffs(pebble->pgp_data->buffer_mask) - 1;
            Transition* event_header = reinterpret_cast<Transition*>(pebble->pgp_data->buffers[index]->virt);
            TransitionId::Value transition_id = event_header->seq.service();

            if (event_header->seq.pulseId().value() != result->seq.pulseId().value()) {
                printf("crap timestamps dont match\n");
            }

            // write event to file if it passes event builder or is a configure transition
            if (eb_decision == 1 || (transition_id == 2)) {
                Dgram* dgram = (Dgram*)pebble->fex_data();
                if (fwrite(dgram, sizeof(Dgram) + dgram->xtc.sizeofPayload(), 1, xtcFile) != 1) {
                    printf("Error writing to output xtc file.\n");
                    return;
                }
            }

            // return buffer to memory pool
            for (int l=0; l<8; l++) {
                if (pebble->pgp_data->buffer_mask & (1 << l)) {
                    pool.dma.buffer_queue.push(pebble->pgp_data->buffers[l]);
                }
            }
            pebble->pgp_data->counter = 0;
            pebble->pgp_data->buffer_mask = 0;
            pool.pebble_queue.push(pebble);

            result = (Dgram*)result->xtc.next();
        }
        myBatchMan.inflight_count.fetch_sub(1);
        delete input;
    }
}

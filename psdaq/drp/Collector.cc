#include "Collector.hh"

#include <rdma/fi_domain.h>
#include <thread>
#include <cassert>
#include <unistd.h>                     // sysconf
#include <stdlib.h>                     // posix_memalign
#include <stdio.h>                      // perror
using namespace XtcData;
using namespace Pds::Eb;

MyDgram::MyDgram(unsigned pulseId, uint64_t val, unsigned contributor_id)
{
    seq = XtcData::Sequence(Sequence::Event, TransitionId::L1Accept, TimeStamp(), PulseId(pulseId));
    env = 0;
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

void collector(MemPool& pool, Parameters& para)
{
    Pds::StringList peers;
    peers.push_back(para.eb_server_ip); //acc04 172.21.52.129
    Pds::StringList ports;
    ports.push_back("32768");
    Pds::Eb::EbLfClient myEbLfClient(peers, ports);

    MyBatchManager myBatchMan(myEbLfClient, para.contributor_id);

    unsigned timeout = 120;
    myEbLfClient.connect(para.contributor_id, timeout, myBatchMan.batchRegion(), myBatchMan.batchRegionSize());

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
        // printf("Collector:  Transition id %d pulse id %lu event counter %u \n",
        //        transition_id, event_header->seq.pulseId().value(), event_header->evtCounter);


        // Dgram& dgram = *reinterpret_cast<Dgram*>(pebble->fex_data());
        uint64_t val;
        if (i%3==0) {
            val = 0xdeadbeef;
        } else {
            val = 0xabadcafe;
        }
        MyDgram dg(i, val, para.contributor_id);
        myBatchMan.process(&dg);
        pool.output_queue.push(pebble);
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
    unsigned none = 0;
    unsigned nzero = 0;
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
            // printf("--- result %lx\n",*(uint64_t*)(result->xtc.payload()));
            uint64_t val = *(uint64_t*)(result->xtc.payload());

            Pebble* pebble;
            pool.output_queue.pop(pebble);

            // perform file writing here
            // usleep(10);

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
            if (val==0) {
                nzero++;
            } else if (val==1) {
                none++;
            } else {
                printf("error %ld\n",val);
            }
            // if (nreceive%10000==0) printf("%d %d %d\n", nreceive, none, nzero);
        }
        delete input;
    }
}

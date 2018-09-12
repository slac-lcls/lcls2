#include <linux/limits.h>
#include <thread>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <zmq.h>
#include "Collector.hh"
#include "psdaq/eb/utilities.hh"
#include "psdaq/eb/EbContributor.hh"
using namespace XtcData;
using namespace Pds::Eb;

MyDgram::MyDgram(Sequence& sequence, uint64_t val, unsigned contributor_id)
{
    seq = sequence;
    xtc = Xtc(TypeId(TypeId::Data, 0), TheSrc(Level::Segment, contributor_id));
    _data = val;
    xtc.alloc(sizeof(_data));
}


EbReceiver::EbReceiver(const Parameters& para,
                       MemPool&          pool,
                       MonContributor*   mon) :
  EbCtrbInBase(para.tPrms),
      _pool(pool),
      _xtcFile(nullptr),
      _mon(mon),
      nreceive(0)
{
    char file_name[PATH_MAX];
    snprintf(file_name, PATH_MAX, "%s/data-%02d.xtc", para.output_dir.c_str(), para.tPrms.id);
    FILE* xtcFile = fopen(file_name, "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
        return;
    }
    _xtcFile = xtcFile;
}

void EbReceiver::process(const Dgram* result, const void* appPrm)
{
    nreceive++;
    uint64_t eb_decision = *(uint64_t*)(result->xtc.payload());
    // printf("eb decision %lu\n", eb_decision);
    Pebble* pebble = (Pebble*)appPrm;

    int index = __builtin_ffs(pebble->pgp_data->buffer_mask) - 1;
    Transition* event_header = reinterpret_cast<Transition*>(pebble->pgp_data->buffers[index]->virt);
    TransitionId::Value transition_id = event_header->seq.service();

    if (event_header->seq.pulseId().value() != result->seq.pulseId().value()) {
        printf("crap timestamps dont match\n");
    }

    // write event to file if it passes event builder or is a configure transition
    if (eb_decision == 1 || (transition_id == 2)) {
        Dgram* dgram = (Dgram*)pebble->fex_data();
        if (fwrite(dgram, sizeof(Dgram) + dgram->xtc.sizeofPayload(), 1, _xtcFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return;
        }
    }

    if (_mon) {
        Dgram* dgram = (Dgram*)pebble->fex_data();
        if (result->seq.isEvent()) {    // L1Accept
            uint32_t* response = (uint32_t*)result->xtc.payload();

            if (response[1])  _mon->post(dgram, response[1]);
        } else {                        // Other Transition
            _mon->post(dgram);
        }
    }

    // return buffer to memory pool
    for (int l=0; l<8; l++) {
        if (pebble->pgp_data->buffer_mask & (1 << l)) {
            _pool.dma.buffer_queue.push(pebble->pgp_data->buffers[l]);
        }
    }
    pebble->pgp_data->counter = 0;
    pebble->pgp_data->buffer_mask = 0;
    _pool.pebble_queue.push(pebble);
}

// collects events from the workers and sends them to the event builder
void collector(MemPool& pool, Parameters& para, EbContributor& ebCtrb, MonContributor* meb)
{
    void* context = zmq_ctx_new();
    void* socket = zmq_socket(context, ZMQ_PUSH);
    zmq_connect(socket, "tcp://localhost:5559");

    printf("*** myEb %p %zd\n",ebCtrb.batchRegion(), ebCtrb.batchRegionSize());
    EbReceiver eb_rcvr(para, pool, meb);

    // Wait a bit to allow other components of the system to establish connections
    // Revisit: This should be replaced with some sort of gate that guards
    //          against continuing until all components have established
    //          connections and are ready to take data
    sleep(1);

    // start eb receiver thread
    ebCtrb.startup(eb_rcvr);

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
        MyDgram dg(dgram.seq, val, para.tPrms.id);
        /*
        Dgram dg(dgram);
        dg.xtc.src = TheSrc(Level::Segment, para.contributor_id);
        dg.xtc.extent = sizeof(dg.xtc);
        uint64_t* payload = (uint64_t*)dg.xtc.alloc(8);
        *payload = val;
        */
        ebCtrb.process(&dg, (const void*)pebble);
        i++;
    }

    ebCtrb.shutdown();
}

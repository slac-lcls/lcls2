#include <linux/limits.h>
#include <thread>
#include <unistd.h>
#include <limits.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <zmq.h>
#include "AxisDriver.h"
#include "json.hpp"


#include "Collector.hh"
#include "psdaq/eb/utilities.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/MebContributor.hh"
#include "psdaq/service/Collection.hh"

using json = nlohmann::json;
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
                       MebContributor*   mon) :
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
    Transition* event_header = reinterpret_cast<Transition*>(pebble->pgp_data->buffers[index].data);
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
            dmaRetIndex(_pool.fd, pebble->pgp_data->buffers[l].dmaIndex);
        }
    }
    pebble->pgp_data->counter = 0;
    pebble->pgp_data->buffer_mask = 0;
    _pool.pebble_queue.push(pebble);
}

// collects events from the workers and sends them to the event builder
void collector(MemPool& pool, Parameters& para, TebContributor& ebCtrb, MebContributor* meb)
{
    enum { PORT_BASE = 29980 };         // TODO move to header file
    void* context = zmq_ctx_new();
    void* socket = zmq_socket(context, ZMQ_PUSH);
    char socket_name[PATH_MAX];
    snprintf(socket_name, PATH_MAX, "tcp://%s:%d", para.collect_host.c_str(), PORT_BASE + para.partition);
    if (zmq_connect(socket, socket_name) == -1) {
        perror("ZMQ_PUSH: zmq_connect");
    } else {
        printf("ZMQ_PUSH: zmq_connect(\"%s\")\n", socket_name);
    }

    // generate sender_id
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    int pid = getpid();
    size_t sender_id = std::hash<std::string>{}(std::string(hostname) + std::to_string(pid));

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
        Transition* event_header = reinterpret_cast<Transition*>(pebble->pgp_data->buffers[index].data);
        TransitionId::Value transition_id = event_header->seq.service();
         if (transition_id == 2) {
            printf("Collector saw configure transition\n");
        } else if (transition_id != 0) {
            printf("Collector saw transition ID %d\n", (int)transition_id);
        }
        // pass non L1 accepts to control level
        if (transition_id != 0) {
            char msg_id_buf[32];
            sprintf(msg_id_buf, "%010u-%09u", event_header->seq.stamp().seconds(),
                    event_header->seq.stamp().nanoseconds());
            json body = json({});
            json reply = create_msg("drp-transition", msg_id_buf, sender_id, body);
            std::string s = reply.dump();
            if (zmq_send(socket, s.c_str(), s.length(), 0) == -1) {
                perror("zmq_send");
            } else {
                printf("Send JSON transition over zeromq socket\n");
            }
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

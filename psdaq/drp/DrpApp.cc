#include <iostream>
#include <limits.h>
#include "AreaDetector.hh"
#include "TimingHeader.hh"
#include "Digitizer.hh"
#include "DrpApp.hh"
#include "AxisDriver.h"
#include "xtcdata/xtc/TransitionId.hh"

using namespace Pds::Eb;

DrpApp::DrpApp(Parameters* para) :
    CollectionApp(para->collect_host, para->partition, "drp"),
    m_para(para),
    m_inprocRecv(&m_context, ZMQ_PAIR),
    m_pool(para->numWorkers, para->numEntries)
{
    size_t maxSize = sizeof(MyDgram);
    m_para->tPrms = { /* .ifAddr        = */ { }, // Network interface to use
                      /* .port          = */ { }, // Port served to TEBs
                      /* .partition     = */ m_para->partition,
                      /* .id            = */ 0,
                      /* .builders      = */ 0,   // TEBs
                      /* .addrs         = */ { },
                      /* .ports         = */ { },
                      /* .duration      = */ BATCH_DURATION,
                      /* .maxBatches    = */ MAX_BATCHES,
                      /* .maxEntries    = */ MAX_ENTRIES,
                      /* .maxInputSize  = */ maxSize,
                      /* .core          = */ { 11, 12 },
                      /* .verbose       = */ 0 };

    m_para->mPrms = { /* .addrs         = */ { },
                      /* .ports         = */ { },
                      /* .id            = */ 0,
                      /* .maxEvents     = */ 8,    //mon_buf_cnt,
                      /* .maxEvSize     = */ 1024, //mon_buf_size,
                      /* .maxTrSize     = */ 1024, //mon_trSize,
                      /* .verbose       = */ 0 };

    m_ebContributor = std::make_unique<Pds::Eb::TebContributor>(m_para->tPrms);

    m_inprocRecv.bind("inproc://drp");
    std::cout<<"Constructor"<<std::endl;
}

void DrpApp::handleConnect(const json &msg)
{
    parseConnectionParams(msg["body"]);

    // should move into constructor
    Factory<Detector> f;
    f.register_type<Digitizer>("Digitizer");
    f.register_type<AreaDetector>("AreaDetector");
    Detector* det = f.create(m_para->detectorType);
    if (det == nullptr) {
        std::cout<< "Error !! Could not create Detector object\n";
    }

    int laneMask = 0xf;
    m_pgpReader = std::make_unique<PGPReader>(m_pool, det, laneMask, m_para->numWorkers);
    m_pgpThread = std::thread{&PGPReader::run, std::ref(*m_pgpReader)};

    // Create all the eb things and do the connections
    bool connected = true;
    // Pds::Eb::TebContributor ebCtrb(m_para->tPrms);
    int rc = m_ebContributor->connect(m_para->tPrms);
    if (rc) {
        connected = false;
        std::cout<<"TebContributor connect failed\n";
    }

    Pds::Eb::MebContributor* meb = nullptr;
    if (m_para->mPrms.addrs.size() != 0) {
        meb = new Pds::Eb::MebContributor(m_para->mPrms);
        rc = meb->connect(m_para->mPrms);
        if (rc) {
            connected = false;
            std::cout<<"MebContributor connect failed\n";
        }
    }

    m_ebRecv = std::make_unique<EbReceiver>(*m_para, m_pool, meb);
    rc = m_ebRecv->connect(m_para->tPrms);
    if (rc) {
        connected = false;
        std::cout<<"EbReceiver connect failed\n";
    }

    // start performance monitor thread
    m_monitorThread =std::thread(monitor_func,
                               std::ref(*m_para),
                               std::ref(m_pgpReader->get_counters()),
                               std::ref(m_pool),
                               std::ref(*m_ebContributor));

    m_collectorThread = std::thread(&DrpApp::collector, std::ref(*this));

    // reply to collection with connect status
    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    if (connected) {
        reply(answer);
    }
}

void DrpApp::handleConfigure(const json &msg)
{
    int ret = m_inprocRecv.poll(ZMQ_POLLIN, 5000);
    //m_recv.recv
}

void DrpApp::handleReset(const json &msg)
{
}

void DrpApp::parseConnectionParams(const json& body)
{
    std::string id = std::to_string(getId());
    m_para->tPrms.id = body["drp"][id]["drp_id"];
    const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_MEBS + MAX_MEBS;
    const unsigned tebPortBase = TEB_PORT_BASE + numPorts * m_para->partition;
    const unsigned drpPortBase = DRP_PORT_BASE + numPorts * m_para->partition;
    const unsigned mebPortBase = MEB_PORT_BASE + numPorts * m_para->partition;

    m_para->tPrms.port = std::to_string(drpPortBase + m_para->tPrms.id);
    m_para->mPrms.id = m_para->tPrms.id;
    m_para->tPrms.ifAddr = body["drp"][id]["connect_info"]["nic_ip"];

    uint64_t builders = 0;
    for (auto it : body["teb"].items()) {
        unsigned tebId = it.value()["teb_id"];
        std::string address = it.value()["connect_info"]["nic_ip"];
        std::cout << "TEB: " << tebId << "  " << address << '\n';
        builders |= 1ul << tebId;
        m_para->tPrms.addrs.push_back(address);
        m_para->tPrms.ports.push_back(std::string(std::to_string(tebPortBase + tebId)));
    }
    m_para->tPrms.builders = builders;

    if (body.find("meb") != body.end()) {
        for (auto it : body["meb"].items()) {
            unsigned mebId = it.value()["meb_id"];
            std::string address = it.value()["connect_info"]["nic_ip"];
            std::cout << "MEB: " << mebId << "  " << address << '\n';
            m_para->mPrms.addrs.push_back(address);
            m_para->mPrms.ports.push_back(std::string(std::to_string(mebPortBase + mebId)));
        }
    }
}

// collects events from the workers and sends them to the event builder
void DrpApp::collector()
{
    printf("*** myEb %p %zd\n", m_ebContributor->batchRegion(), m_ebContributor->batchRegionSize());

    ZmqSocket inprocSend = ZmqSocket(&m_context, ZMQ_PAIR);
    inprocSend.connect("inproc://drp");

    // start eb receiver thread
    m_ebContributor->startup(*m_ebRecv);

    int i = 0;
    while (true) {
        int worker;
        m_pool.collector_queue.pop(worker);
        Pebble* pebble;
        m_pool.worker_output_queues[worker].pop(pebble);

        int index = __builtin_ffs(pebble->pgp_data->buffer_mask) - 1;
        Pds::TimingHeader* event_header = reinterpret_cast<Pds::TimingHeader*>(pebble->pgp_data->buffers[index].data);
        XtcData::TransitionId::Value transition_id = event_header->seq.service();
        switch (transition_id) {
            case XtcData::TransitionId::Configure:
                printf("Collector saw Configure transition\n");
                break;
            case XtcData::TransitionId::Unconfigure:
                printf("Collector saw Unconfigure transition\n");
                break;
            case XtcData::TransitionId::Enable:
                printf("Collector saw Enable transition\n");
                break;
            case XtcData::TransitionId::Disable:
                printf("Collector saw Disable transition\n");
                break;
            case XtcData::TransitionId::ConfigUpdate:
                printf("Collector saw ConfigUpdate transition\n");
                break;
            case XtcData::TransitionId::BeginRecord:
                printf("Collector saw BeginRecord transition\n");
                break;
            case XtcData::TransitionId::EndRecord:
                printf("Collector saw EndRecord transition\n");
                break;
            default:
                break;
        }
        // pass non L1 accepts to control level
        if (transition_id != XtcData::TransitionId::L1Accept) {
            inprocSend.send("{}");
        }

        XtcData::Dgram& dgram = *reinterpret_cast<XtcData::Dgram*>(pebble->fex_data());
        uint64_t val;
        if (i%5 == 0) {
            val = 0xdeadbeef;
        } else {
            val = 0xabadcafe;
        }
        MyDgram dg(dgram.seq, val, m_para->tPrms.id);
        m_ebContributor->process(&dg, (const void*)pebble);
        i++;
    }

    m_ebContributor->shutdown();
}

MyDgram::MyDgram(XtcData::Sequence& sequence, uint64_t val, unsigned contributor_id)
{
    seq = sequence;
    xtc = XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::Data, 0), XtcData::Src(contributor_id));
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

void EbReceiver::process(const XtcData::Dgram* result, const void* appPrm)
{
    nreceive++;
    uint64_t eb_decision = *(uint64_t*)(result->xtc.payload());
    // printf("eb decision %lu\n", eb_decision);
    Pebble* pebble = (Pebble*)appPrm;

    int index = __builtin_ffs(pebble->pgp_data->buffer_mask) - 1;
    Pds::TimingHeader* event_header = reinterpret_cast<Pds::TimingHeader*>(pebble->pgp_data->buffers[index].data);
    XtcData::TransitionId::Value transition_id = event_header->seq.service();

    if (event_header->seq.pulseId().value() != result->seq.pulseId().value()) {
        printf("crap timestamps dont match\n");
    }

    // write event to file if it passes event builder or is a configure transition
    /* FIXME disable writing for now
    if (eb_decision == 1 || (transition_id == TransitionId::Configure)) {
        Dgram* dgram = (Dgram*)pebble->fex_data();
        if (fwrite(dgram, sizeof(Dgram) + dgram->xtc.sizeofPayload(), 1, _xtcFile) != 1) {
            printf("Error writing to output xtc file.\n");
            return;
        }
    }
    */
    if (_mon) {
        XtcData::Dgram* dgram = (XtcData::Dgram*)pebble->fex_data();
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

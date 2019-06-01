#include <iostream>
#include <fstream>
#include <limits.h>
#include "TimingSystem.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "TimingHeader.hh"
#include "AxisDriver.h"
#include "xtcdata/xtc/TransitionId.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "DrpApp.hh"

using namespace Pds::Eb;

using json = nlohmann::json;

namespace Drp {

DrpApp::DrpApp(Parameters* para) :
    CollectionApp(para->collectionHost, para->partition, "drp", para->detName+std::to_string(para->detSegment)),
    m_para(para),
    m_pool(*para)
{
    size_t maxSize = sizeof(MyDgram);
    m_tPrms = { /* .ifAddr        = */ { }, // Network interface to use
                      /* .port          = */ { }, // Port served to TEBs
                      /* .partition     = */ m_para->partition,
                      /* .alias         = */ { }, // Unique name from cmd line
                      /* .id            = */ 0,
                      /* .builders      = */ 0,   // TEBs
                      /* .addrs         = */ { },
                      /* .ports         = */ { },
                      /* .maxInputSize  = */ maxSize,
                      /* .core          = */ { 11, 12 },
                      /* .verbose       = */ 0,
                      /* .readoutGroup  = */ 0,
                      /* .contractor    = */ 0 };

    m_mPrms = { /* .addrs         = */ { },
                      /* .ports         = */ { },
                      /* .partition     = */ m_para->partition,
                      /* .id            = */ 0,
                      /* .maxEvents     = */ 8,    //mon_buf_cnt,
                      /* .maxEvSize     = */ 65536, //mon_buf_size,
                      /* .maxTrSize     = */ 65536, //mon_trSize,
                      /* .verbose       = */ 0 };

    try {
        m_exposer = std::make_unique<prometheus::Exposer>("0.0.0.0:9200", "/metrics", 1);
    }
    catch(const std::runtime_error& e) {
        std::cout<<"Could not start monitoring server!!\n";
        std::cout<<e.what()<<std::endl;
    }
    Factory<Detector> f;
    f.register_type<TimingSystem>("TimingSystem");
    f.register_type<Digitizer>("Digitizer");
    f.register_type<AreaDetector>("AreaDetector");
    m_det = f.create(m_para, &m_pool);
    if (m_det == nullptr) {
        std::cout<< "Error !! Could not create Detector object\n";
    }
    std::cout << "output dir: " << m_para->outputDir << std::endl;
}

json DrpApp::connectionInfo()
{
    std::string ip = getNicIp();
    std::cout<<"nic ip  "<<ip<<'\n';
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    return body;
}

void DrpApp::handleConnect(const json &msg)
{
    parseConnectionParams(msg["body"]);

    m_det->nodeId = m_tPrms.id;
    m_det->connect(msg);

    auto exporter = std::make_shared<MetricExporter>();
    if (m_exposer) {
        m_exposer->RegisterCollectable(exporter);
    }
    m_pgpReader = std::make_unique<PGPReader>(*m_para, m_pool, m_det);
    m_pgpThread = std::thread{&PGPReader::run, std::ref(*m_pgpReader), exporter};

    // Create all the eb things and do the connections
    bool connected = true;
    m_ebContributor = std::make_unique<TebContributor>(m_tPrms, exporter);
    int rc = m_ebContributor->connect(m_tPrms);
    if (rc) {
        connected = false;
        std::cout<<"TebContributor connect failed\n";
    }

    if (m_mPrms.addrs.size() != 0) {
        m_meb = std::make_unique<MebContributor>(m_mPrms, exporter);
        void* poolBase = (void*)m_pool.pebble[0];
        size_t poolSize = m_pool.pebble.size();
        rc = m_meb->connect(m_mPrms, poolBase, poolSize);
        if (rc) {
            connected = false;
            std::cout<<"MebContributor connect failed\n";
        }
    }

    m_ebRecv = std::make_unique<EbReceiver>(*m_para, m_tPrms, m_pool, context(), m_meb.get(), exporter);
    rc = m_ebRecv->connect(m_tPrms);
    if (rc) {
        connected = false;
        std::cout<<"EbReceiver connect failed\n";
    }

    m_collectorThread = std::thread(&DrpApp::collector, std::ref(*this));

    // reply to collection with connect status
    json body = json({});
    if (!connected) {
        body["error"] = "connect error";
        std::cout<<"connect error\n";
    }
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void DrpApp::handlePhase1(const json &msg)
{
    std::cout<<"handlePhase1 in DrpApp\n";

    std::string key = msg["header"]["key"];
    unsigned error = 0;
    if (key == "configure") {
        XtcData::Xtc& xtc = m_det->transitionXtc();
        XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
        xtc.contains = tid;
        xtc.damage = 0;
        xtc.extent = sizeof(XtcData::Xtc);
        error = m_det->configure(xtc);
    }

    json answer;
    json body = json({});
    if (error) {
        body["error"] = "phase 1 error";
        std::cout<<"transition phase1 error\n";
    }
    else {
        std::cout<<"transition phase1 complete\n";
    }
    answer = createMsg(msg["header"]["key"], msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void DrpApp::handleReset(const json &msg)
{
}

void DrpApp::parseConnectionParams(const json& body)
{
    std::string id = std::to_string(getId());
    m_tPrms.id = body["drp"][id]["drp_id"];
    const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_MEBS + MAX_MEBS;
    const unsigned tebPortBase = TEB_PORT_BASE + numPorts * m_para->partition;
    const unsigned drpPortBase = DRP_PORT_BASE + numPorts * m_para->partition;
    const unsigned mebPortBase = MEB_PORT_BASE + numPorts * m_para->partition;

    m_tPrms.port = std::to_string(drpPortBase + m_tPrms.id);
    m_mPrms.id = m_tPrms.id;
    m_tPrms.ifAddr = body["drp"][id]["connect_info"]["nic_ip"];

    uint64_t builders = 0;
    for (auto it : body["teb"].items()) {
        unsigned tebId = it.value()["teb_id"];
        std::string address = it.value()["connect_info"]["nic_ip"];
        std::cout << "TEB: " << tebId << "  " << address << '\n';
        builders |= 1ul << tebId;
        m_tPrms.addrs.push_back(address);
        m_tPrms.ports.push_back(std::string(std::to_string(tebPortBase + tebId)));
    }
    m_tPrms.builders = builders;

    m_tPrms.readoutGroup = 1 << unsigned(body["drp"][id]["det_info"]["readout"]);
    m_tPrms.contractor = m_tPrms.readoutGroup; // Revisit: Value to come from CfgDb

    if (body.find("meb") != body.end()) {
        for (auto it : body["meb"].items()) {
            unsigned mebId = it.value()["meb_id"];
            std::string address = it.value()["connect_info"]["nic_ip"];
            std::cout << "MEB: " << mebId << "  " << address << '\n';
            m_mPrms.addrs.push_back(address);
            m_mPrms.ports.push_back(std::string(std::to_string(mebPortBase + mebId)));
        }
    }
}

// collects events from the workers and sends them to the event builder
void DrpApp::collector()
{
    // start eb receiver thread
    m_ebContributor->startup(*m_ebRecv);

    int64_t worker = 0L;
    int64_t counter = 0L;
    Batch batch;
    while (true) {
        if (!m_pool.workerOutputQueues[worker % m_para->nworkers].pop(batch)) {
            break;
        }
        //std::cout<<"collector:  "<<batch.start<<"  "<<batch.end<<'\n';
        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) % m_pool.nbuffers;
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
            void* buffer = m_ebContributor->allocate(dgram, (void*)((uintptr_t)index));
            if (buffer) // else this DRP doesn't provide input, or timed out
            {
                MyDgram* dg = new(buffer) MyDgram(*dgram, val, m_tPrms.id);
                m_ebContributor->process(dg);
            }
            counter++;
        }
        worker++;
    }
    m_ebContributor->shutdown();
}

void DrpApp::shutdown()
{
    if (m_pgpReader) {
        m_pgpReader->shutdown();
        m_pgpThread.join();
        for (unsigned i = 0; i < m_para->nworkers; i++) {
            m_pool.workerOutputQueues[i].shutdown();
        }
        m_collectorThread.join();
    }
}

MyDgram::MyDgram(XtcData::Dgram& dgram, uint64_t val, unsigned contributor_id)
{
    seq = dgram.seq;
    env = dgram.env;
    xtc = XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::Data, 0), XtcData::Src(contributor_id));
    _data = val;
    xtc.alloc(sizeof(_data));
}


EbReceiver::EbReceiver(const Parameters& para, Pds::Eb::TebCtrbParams& tPrms,
                       MemPool& pool, ZmqContext& context, MebContributor* mon,
                       std::shared_ptr<MetricExporter> exporter) :
  EbCtrbInBase(tPrms, exporter),
  m_pool(pool),
  m_mon(mon),
  m_fileWriter(4194304),
  m_smdWriter(1048576),
  m_inprocSend(&context, ZMQ_PAIR),
  m_count(0),
  m_offset(0)
{
    m_inprocSend.connect("inproc://drp");

    if (!para.outputDir.empty()) {
        std::string fileName = {para.outputDir + "/data-" + std::to_string(tPrms.id) + ".xtc2"};
        // cpo suggests leaving this print statement in because
        // filesystems can hang in ways we can't timeout/detect
        // and this print statement may speed up debugging significantly.
        std::cout << "Opening file " << fileName << std::endl;
        m_fileWriter.open(fileName);
        m_smdWriter.open({para.outputDir + "/data-" + std::to_string(tPrms.id) + "smd.xtc2"});
        m_writing = true;
    }
    else {
        m_writing = false;
    }
    m_nodeId = tPrms.id;
}

void EbReceiver::process(const XtcData::Dgram* result, const void* appPrm)
{
    uint32_t* ebDecision = (uint32_t*)(result->xtc.payload());
    //printf("eb decisions write: %u, monitor: %u\n", ebDecision[WRT_IDX], ebDecision[MON_IDX]);
    unsigned index = (uintptr_t)appPrm;
    PGPEvent* event = &m_pool.pgpEvents[index];

    // get transitionId from the first lane in the event
    int lane = __builtin_ffs(event->mask) - 1;
    uint32_t dmaIndex = event->buffers[lane].index;
    Pds::TimingHeader* timingHeader = (Pds::TimingHeader*)m_pool.dmaBuffers[dmaIndex];
    XtcData::TransitionId::Value transitionId = timingHeader->seq.service();
    //printf("EbReceiver:  %u   index %u\n", timingHeader->evtCounter, index);

    // pass non L1 accepts to control level
    if (transitionId != XtcData::TransitionId::L1Accept) {
        // send pulseId to inproc so it gets forwarded to the collection
        m_inprocSend.send(std::to_string(timingHeader->seq.pulseId().value()));
        printf("EbReceiver saw %s transition\n", XtcData::TransitionId::name(transitionId));
    }

    if (index != ((lastIndex + 1) & (m_pool.nbuffers - 1))) {
        printf("\033[0;31m");
        printf("jumping index %u  previous index %u\n", index, lastIndex);
        printf("evtCounter %u\n", timingHeader->evtCounter);
        printf("lastevtCounter %u\n", lastEvtCounter);
        printf("\033[0m");
    }

    if (timingHeader->seq.pulseId().value() != result->seq.pulseId().value()) {
        std::cout<<"crap timestamps dont match\n";
        printf("index %u  previous index %u\n", index, lastIndex);
        std::cout<<"pebble pulseId  "<<timingHeader->seq.pulseId().value()<<
                 "  result dgram pulseId  "<<result->seq.pulseId().value()<<'\n';
        exit(-1);
    }

    lastIndex = index;
    lastEvtCounter = timingHeader->evtCounter;

    XtcData::Dgram* dgram = (XtcData::Dgram*)m_pool.pebble[index];
    // write event to file if it passes event builder or is a configure transition
    if (m_writing) {
        if (ebDecision[WRT_IDX] == 1 || (transitionId == XtcData::TransitionId::Configure)) {
            size_t size = sizeof(XtcData::Dgram) + dgram->xtc.sizeofPayload();
            m_fileWriter.writeEvent(dgram, size);

            // small data writing
            XtcData::Dgram& smdDgram = *(XtcData::Dgram*)m_smdWriter.buffer;
            smdDgram.seq = dgram->seq;
            XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
            smdDgram.xtc.contains = tid;
            smdDgram.xtc.damage = 0;
            smdDgram.xtc.extent = sizeof(XtcData::Xtc);

            if (transitionId == XtcData::TransitionId::Configure) {
                m_smdWriter.addNames(smdDgram.xtc, m_nodeId);
            }

            XtcData::NamesId namesId(m_nodeId, 0);
            XtcData::CreateData smd(smdDgram.xtc, m_smdWriter.namesLookup, namesId);
            smd.set_value(SmdDef::intOffset, m_offset);
            smd.set_value(SmdDef::intDgramSize, size);
            m_smdWriter.writeEvent(&smdDgram, sizeof(XtcData::Dgram) + smdDgram.xtc.sizeofPayload());

            m_offset += size;
        }
    }

    if (m_mon) {
        // L1Accept
        if (result->seq.isEvent()) {
            if (ebDecision[MON_IDX])  m_mon->post(dgram, ebDecision[MON_IDX]);
        }
        // Other Transition
        else {
            m_mon->post(dgram);
        }
    }

    // return buffers and reset event
    for (int i=0; i<4; i++) {
        if (event->mask & (1 << i)) {
            m_indices[m_count] = event->buffers[i].index;
            m_count++;
            if (m_count == m_size) {
                dmaRetIndexes(m_pool.fd, m_count, m_indices);
                // std::cout<<"return dma buffers to driver\n";
                m_count = 0;
            }
        }
    }
    event->mask = 0;
}

}

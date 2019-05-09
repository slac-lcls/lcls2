#include <iostream>
#include <fstream>
#include <limits.h>
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "TimingHeader.hh"
#include "AxisDriver.h"
#include "xtcdata/xtc/TransitionId.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "DrpApp.hh"

static const unsigned RTMON_RATE = 1;    // Publish rate in seconds
static const unsigned RTMON_VERBOSE = 0;

using namespace Pds::Eb;

namespace Drp {

DrpApp::DrpApp(Parameters* para) :
    CollectionApp(para->collectionHost, para->partition, "drp", para->alias),
    m_para(para),
    m_pool(*para),
    m_smon("psmetric04", RTMON_PORT_BASE, m_para->partition, RTMON_RATE, RTMON_VERBOSE)
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
                      /* .id            = */ 0,
                      /* .maxEvents     = */ 8,    //mon_buf_cnt,
                      /* .maxEvSize     = */ 65536, //mon_buf_size,
                      /* .maxTrSize     = */ 65536, //mon_trSize,
                      /* .verbose       = */ 0 };

    m_ebContributor = std::make_unique<TebContributor>(m_tPrms, m_smon);
    std::cout << "output dir: " << m_para->outputDir << std::endl;
}

void DrpApp::handleConnect(const json &msg)
{
    parseConnectionParams(msg["body"]);

    // should move into constructor
    Factory<Detector> f;
    f.register_type<Digitizer>("Digitizer");
    f.register_type<AreaDetector>("AreaDetector");
    std::cout<<"nodeId  "<<m_tPrms.id<<'\n';
    m_det = f.create(m_para, &m_pool, m_tPrms.id);
    if (m_det == nullptr) {
        std::cout<< "Error !! Could not create Detector object\n";
    }
    m_det->connect();

    m_pgpReader = std::make_unique<PGPReader>(*m_para, m_pool, m_det);
    m_pgpThread = std::thread{&PGPReader::run, std::ref(*m_pgpReader)};

    // Create all the eb things and do the connections
    bool connected = true;
    int rc = m_ebContributor->connect(m_tPrms);
    if (rc) {
        connected = false;
        std::cout<<"TebContributor connect failed\n";
    }

    if (m_mPrms.addrs.size() != 0) {
        m_meb = std::make_unique<MebContributor>(m_mPrms, m_smon);
        void* poolBase = (void*)m_pool.pebble[0];
        size_t poolSize = m_pool.pebble.size();
        rc = m_meb->connect(m_mPrms, poolBase, poolSize);
        if (rc) {
            connected = false;
            std::cout<<"MebContributor connect failed\n";
        }
    }

    m_ebRecv = std::make_unique<EbReceiver>(*m_para, m_tPrms, m_pool, context(), m_meb.get(), m_smon);
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
        error = m_det->configure(m_det->transitionXtc());
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
                       StatsMonitor& smon) :
  EbCtrbInBase(tPrms, smon),
  m_pool(pool),
  m_mon(mon),
  m_inprocSend(&context, ZMQ_PAIR),
  m_count(0)
{
    m_inprocSend.connect("inproc://drp");

    if (!para.outputDir.empty()) {
        std::string fileName = {para.outputDir + "/data-" + std::to_string(tPrms.id) + ".xtc2"};
        // cpo suggests leaving this print statement in because
        // filesystems can hang in ways we can't timeout/detect
        // and this print statement may speed up debugging significantly.
        std::cout << "Opening file " << fileName << std::endl;
        m_fileWriter.open(fileName);
        m_writing = true;
    }
    else {
        m_writing = false;
    }
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

BufferedFileWriter::BufferedFileWriter() :
    m_count(0), m_buffer(BufferSize)
{
}

BufferedFileWriter::~BufferedFileWriter()
{
    write(m_fd, m_buffer.data(), m_count);
    m_count = 0;
}

void BufferedFileWriter::open(std::string& fileName)
{
    m_fd = ::open(fileName.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
    if (m_fd == -1) {
        std::cout<<"Error creating file "<<fileName<<'\n';
    }
}

void BufferedFileWriter::writeEvent(void* data, size_t size)
{
    // doesn't fit into the remaing m_buffer
    if (size > (BufferSize - m_count)) {
        write(m_fd, m_buffer.data(), m_count);
        m_count = 0;
    }
    memcpy(m_buffer.data()+m_count, data, size);
    m_count += size;
}

}

#include <iostream>
#include <limits.h>
#include "AreaDetector.hh"
#include "TimingHeader.hh"
#include "Digitizer.hh"
#include "DrpApp.hh"
#include "AxisDriver.h"
#include "xtcdata/xtc/TransitionId.hh"
#include "xtcdata/xtc/Dgram.hh"

static const unsigned RTMON_RATE = 1;    // Publish rate in seconds
static const unsigned RTMON_VERBOSE = 0;

using namespace Pds::Eb;

DrpApp::DrpApp(Parameters* para) :
    CollectionApp(para->collect_host, para->partition, "drp", para->alias),
    m_para(para),
    m_inprocRecv(&m_context, ZMQ_PAIR),
    m_pool(*para),
    m_smon("psmetric04", RTMON_PORT_BASE, m_para->partition, RTMON_RATE, RTMON_VERBOSE)
{
    size_t maxSize = sizeof(MyDgram);
    m_para->tPrms = { /* .ifAddr        = */ { }, // Network interface to use
                      /* .port          = */ { }, // Port served to TEBs
                      /* .partition     = */ unsigned(m_para->partition),
                      /* .alias         = */ { }, // Unique name from cmd line
                      /* .id            = */ 0,
                      /* .builders      = */ 0,   // TEBs
                      /* .addrs         = */ { },
                      /* .ports         = */ { },
                      /* .maxInputSize  = */ maxSize,
                      /* .core          = */ { 11, 12 },
                      /* .verbose       = */ 0,
                      /* .groups        = */ 0,
                      /* .contractor    = */ 0 };

    m_para->mPrms = { /* .addrs         = */ { },
                      /* .ports         = */ { },
                      /* .id            = */ 0,
                      /* .maxEvents     = */ 8,    //mon_buf_cnt,
                      /* .maxEvSize     = */ 65536, //mon_buf_size,
                      /* .maxTrSize     = */ 65536, //mon_trSize,
                      /* .verbose       = */ 0 };

    m_ebContributor = std::make_unique<TebContributor>(m_para->tPrms, m_smon);

    m_inprocRecv.bind("inproc://drp");

    std::cout << "output dir: " << m_para->output_dir << std::endl;
}

void DrpApp::handleConnect(const json &msg)
{
    parseConnectionParams(msg["body"]);

    // should move into constructor
    Factory<Detector> f;
    f.register_type<Digitizer>("Digitizer");
    f.register_type<AreaDetector>("AreaDetector");
    std::cout<<"nodeId  "<<m_para->tPrms.id<<'\n';
    m_det = f.create(m_para);
    if (m_det == nullptr) {
        std::cout<< "Error !! Could not create Detector object\n";
    }
    m_det->connect();

    m_pgpReader = std::make_unique<PGPReader>(m_pool, *m_para, m_det, m_para->laneMask);
    m_pgpThread = std::thread{&PGPReader::run, std::ref(*m_pgpReader)};

    // Create all the eb things and do the connections
    bool connected = true;
    int rc = m_ebContributor->connect(m_para->tPrms);
    if (rc) {
        connected = false;
        std::cout<<"TebContributor connect failed\n";
    }

    if (m_para->mPrms.addrs.size() != 0) {
        m_meb = std::make_unique<MebContributor>(m_para->mPrms, m_smon);
        void* poolBase = (void*)m_pool.pebble.data();
        size_t poolSize = m_pool.pebble.size() * sizeof(Pebble);
        rc = m_meb->connect(m_para->mPrms, poolBase, poolSize);
        if (rc) {
            connected = false;
            std::cout<<"MebContributor connect failed\n";
        }
    }

    m_ebRecv = std::make_unique<EbReceiver>(*m_para, m_pool, m_context, m_meb.get(), m_smon);
    rc = m_ebRecv->connect(m_para->tPrms);
    if (rc) {
        connected = false;
        std::cout<<"EbReceiver connect failed\n";
    }

    // start performance monitor thread
    m_monitorThread = std::thread(monitor_func,
                               std::ref(*m_para),
                               std::ref(m_pgpReader->get_counters()),
                               std::ref(m_pool),
                               std::ref(*m_ebContributor),
                               std::ref(m_smon));

    m_collectorThread = std::thread(&DrpApp::collector, std::ref(*this));

    // reply to collection with connect status
    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    if (connected) {
        reply(answer);
    }
}

void DrpApp::handlePhase1(const json &msg)
{
    std::cout<<"handle configure DrpApp\n";

    XtcData::Dgram& dgram = m_det->transitionDgram();
    XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
    dgram.xtc.contains = tid;
    dgram.xtc.damage = 0;
    dgram.xtc.extent = sizeof(XtcData::Xtc);

    std::string key = msg["header"]["key"];
    unsigned error=0;
    if (key == "configure1") {
        error = m_det->configure(dgram);
    }
    // check for message from timing system
    json answer;
    json body = json({});
    if (error) {
        body["error"] = "phase 2 error";
        std::cout<<"transition phase1 error\n";
    }
    else {
        std::cout<<"transition phase1 complete\n";
    }
    answer = createMsg(msg["header"]["key"], msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void DrpApp::handlePhase2(const json &msg)
{
    int ret = m_inprocRecv.poll(ZMQ_POLLIN, 5000);
    json answer;
    json body = json({});
    if (ret) {
        json reply = m_inprocRecv.recvJson();
        std::cout<<"inproc message received\n";
    }
    else {
        body["error"] = "phase 2 error";
        std::cout<<"phase 2 error\n";
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

    m_para->tPrms.groups = 1 << m_para->partition; // Revisit: Value to come from CfgDb
    m_para->tPrms.contractor = 1 << m_para->partition;  // Revisit: Value to come from CfgDb

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
    // start eb receiver thread
    m_ebContributor->startup(*m_ebRecv);

    int i = 0;
    while (true) {
        int worker;
        m_pool.collector_queue.pop(worker);
        Pebble* pebble;
        m_pool.worker_output_queues[worker].pop(pebble);

        XtcData::Dgram& dgram = *reinterpret_cast<XtcData::Dgram*>(pebble->fex_data());
        uint64_t val;
        if (i%2 == 0) {
            val = 0xdeadbeef;
        }
        else {
            val = 0xabadcafe;
        }
        // always monitor every event
        val |= 0x1234567800000000ul;
        void* buffer = m_ebContributor->allocate(&dgram, (const void*)pebble);
        if (buffer) // else this DRP doesn't provide input, or timed out
        {
            MyDgram* dg = new(buffer) MyDgram(dgram, val, m_para->tPrms.id);
            m_ebContributor->process(dg);
        }
        i++;
    }

    m_ebContributor->shutdown();
}

MyDgram::MyDgram(XtcData::Dgram& dgram, uint64_t val, unsigned contributor_id)
{
    seq = dgram.seq;
    env = dgram.env;
    xtc = XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::Data, 0), XtcData::Src(contributor_id));
    _data = val;
    xtc.alloc(sizeof(_data));
}


EbReceiver::EbReceiver(const Parameters& para, MemPool& pool,
                       ZmqContext& context, MebContributor* mon,
                       StatsMonitor& smon) :
  EbCtrbInBase(para.tPrms, smon),
  m_pool(pool),
  m_mon(mon),
  nreceive(0),
  m_indexReturner(m_pool.fd),
  m_inprocSend(&context, ZMQ_PAIR)
{
    m_inprocSend.connect("inproc://drp");

    if (!para.output_dir.empty()) {
        std::string fileName = {para.output_dir + "/data-" + std::to_string(para.tPrms.id) + ".xtc2"};
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
    nreceive++;
    uint32_t* eb_decision = (uint32_t*)(result->xtc.payload());
    // printf("eb decisions write: %u, monitor: %u\n", eb_decision[WRT_IDX], eb_decision[MON_IDX]);
    Pebble* pebble = (Pebble*)appPrm;

    int index = __builtin_ffs(pebble->pgp_data->buffer_mask) - 1;
    Pds::TimingHeader* event_header = reinterpret_cast<Pds::TimingHeader*>(pebble->pgp_data->buffers[index].data);
    XtcData::TransitionId::Value transition_id = event_header->seq.service();

    // pass non L1 accepts to control level
    if (transition_id != XtcData::TransitionId::L1Accept) {
        m_inprocSend.send("{}");
        printf("EbReceiver saw %s transition\n", XtcData::TransitionId::name(transition_id));
    }

    if (event_header->seq.pulseId().value() != result->seq.pulseId().value()) {
        std::cout<<"crap timestamps dont match\n";
        std::cout<<"pebble pulseId  "<<event_header->seq.pulseId().value()<<
                 "  result dgram pulseId  "<<result->seq.pulseId().value()<<'\n';
    }

    // write event to file if it passes event builder or is a configure transition
    if (m_writing) {
        if (eb_decision[WRT_IDX] == 1 || (transition_id == XtcData::TransitionId::Configure)) {
            XtcData::Dgram* dgram = (XtcData::Dgram*)pebble->fex_data();
            size_t size = sizeof(XtcData::Dgram) + dgram->xtc.sizeofPayload();
            m_fileWriter.writeEvent(dgram, size);
        }
    }

    if (m_mon) {
        XtcData::Dgram* dgram = (XtcData::Dgram*)pebble->fex_data();
        // L1Accept
        if (result->seq.isEvent()) {
            if (eb_decision[MON_IDX])  m_mon->post(dgram, eb_decision[MON_IDX]);
        }
        // Other Transition
        else {
            m_mon->post(dgram);
        }
    }

    // return buffer to memory pool
    for (int l=0; l<8; l++) {
        if (pebble->pgp_data->buffer_mask & (1 << l)) {
            m_indexReturner.returnIndex(pebble->pgp_data->buffers[l].dmaIndex);
        }
    }
    pebble->pgp_data->counter = 0;
    pebble->pgp_data->buffer_mask = 0;
    m_pool.pebble_queue.push(pebble);
}

DmaIndexReturner::DmaIndexReturner(int fd) : m_fd(fd), m_counts(0) {}

DmaIndexReturner::~DmaIndexReturner()
{
    if (m_counts > 0) {
        dmaRetIndexes(m_fd, m_counts, m_indices);
    }
}

void DmaIndexReturner::returnIndex(uint32_t index)
{
    m_indices[m_counts] = index;
    m_counts++;
    if (m_counts == BatchSize) {
        dmaRetIndexes(m_fd, m_counts, m_indices);
        m_counts = 0;
    }
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

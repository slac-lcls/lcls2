#include <iostream>
#include "TimingHeader.hh"
#include "DataDriver.h"
#include "DrpBase.hh"

using json = nlohmann::json;

namespace Drp {

MyDgram::MyDgram(XtcData::Dgram& dgram, uint64_t val, unsigned contributor_id)
{
    seq = dgram.seq;
    env = dgram.env;
    xtc = XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::Data, 0), XtcData::Src(contributor_id));
    _data = val;
    xtc.alloc(sizeof(_data));
}

unsigned nextPowerOf2(unsigned n)
{
    unsigned count = 0;

    if (n && !(n & (n - 1))) {
        return n;
    }

    while( n != 0) {
        n >>= 1;
        count += 1;
    }

    return 1 << count;
}

MemPool::MemPool(const Parameters& para)
{
    m_fd = open(para.device.c_str(), O_RDWR);
    if (m_fd < 0) {
        std::cout<<"Error opening "<<para.device<<'\n';
        throw "Error opening kcu1500!!\n";
    }

    uint32_t dmaCount;
    dmaBuffers = dmaMapDma(m_fd, &dmaCount, &m_dmaSize);
    if (dmaBuffers == NULL ) {
        std::cout<<"Failed to map dma buffers!\n";
        throw "Error calling dmaMapDma!!\n";
    }
    printf("dmaCount %u  dmaSize %u\n", dmaCount, m_dmaSize);

    // make sure there are more buffers in the pebble than in the pgp driver
    // otherwise the pebble buffers will be overwritten by the pgp event builder
    m_nbuffers = nextPowerOf2(dmaCount);

    // make the size of the pebble buffer that will contain the datagram equal
    // to the dmaSize times the number of lanes
    m_bufferSize = __builtin_popcount(para.laneMask) * m_dmaSize;
    pebble.resize(m_nbuffers, m_bufferSize);
    printf("nbuffer %u  pebble buffer size %u\n", m_nbuffers, m_bufferSize);

    pgpEvents.resize(m_nbuffers);
}


EbReceiver::EbReceiver(const Parameters& para, Pds::Eb::TebCtrbParams& tPrms,
                       MemPool& pool, ZmqSocket& inprocSend, Pds::Eb::MebContributor* mon,
                       std::shared_ptr<MetricExporter>& exporter) :
  EbCtrbInBase(tPrms, exporter),
  m_pool(pool),
  m_mon(mon),
  m_fileWriter(4194304),
  m_smdWriter(1048576),
  m_inprocSend(inprocSend),
  m_count(0),
  m_offset(0)
{
    if (!para.outputDir.empty()) {
        std::string fileName = {para.outputDir + "/data-" + std::to_string(tPrms.id) + ".xtc2"};
        // cpo suggests leaving this print statement in because
        // filesystems can hang in ways we can't timeout/detect
        // and this print statement may speed up debugging significantly.
        std::cout << "Opening file " << fileName << std::endl;
        m_fileWriter.open(fileName);
        m_smdWriter.open({para.outputDir + "/data-" + std::to_string(tPrms.id) + ".smd.xtc2"});
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
    //printf("EbReceiver:  %u   index %u  tid %u  pid %014lx  env %08x\n",
    //       timingHeader->evtCounter, index, transitionId, timingHeader->seq.pulseId().value(), timingHeader->env);

    // pass everything except L1 accepts and slow updates to control level
    if ((transitionId != XtcData::TransitionId::L1Accept)) {
        // send pulseId to inproc so it gets forwarded to the collection
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
            m_inprocSend.send(std::to_string(timingHeader->seq.pulseId().value()));
        }
        printf("EbReceiver saw %s transition @ %014lx\n", XtcData::TransitionId::name(transitionId), timingHeader->seq.pulseId().value());
    }

    if (index != ((lastIndex + 1) & (m_pool.nbuffers() - 1))) {
        printf("\033[0;31m");
        printf("jumping index %u  previous index %u  diff %d\n", index, lastIndex, index - lastIndex);
        printf("evtCounter %u\n", timingHeader->evtCounter);
        printf("pid = %014lx, env = %08x\n", timingHeader->seq.pulseId().value(), timingHeader->env);
        printf("tid %s\n", XtcData::TransitionId::name(transitionId));
        printf("lastevtCounter %u\n", lastEvtCounter);
        printf("lastPid %014lx lastTid %s\n", lastPid, XtcData::TransitionId::name(lastTid));
        printf("\033[0m");
    }

    if (timingHeader->seq.pulseId().value() != result->seq.pulseId().value()) {
        std::cout<<"crap timestamps dont match\n";
        printf("index %u  previous index %u\n", index, lastIndex);
        std::cout<<"pebble pulseId  "<<timingHeader->seq.pulseId().value()<<
                 "  result dgram pulseId  "<<result->seq.pulseId().value()<<'\n';
        uint64_t tPid = timingHeader->seq.pulseId().value();
        uint64_t rPid = result->seq.pulseId().value();
        printf("pebble PID %014lx, result PID %014lx, xor %014lx, diff %ld\n", tPid, rPid, tPid ^ rPid, tPid - rPid);
        exit(-1);
    }

    lastIndex = index;
    lastEvtCounter = timingHeader->evtCounter;
    lastPid = timingHeader->seq.pulseId().value();
    lastTid = timingHeader->seq.service();

    XtcData::Dgram* dgram = (XtcData::Dgram*)m_pool.pebble[index];
    if (m_writing) {
        // write event to file if it passes event builder or if it's a transition
        if ((ebDecision[Pds::Eb::WRT_IDX] == 1) || (transitionId != XtcData::TransitionId::L1Accept)) {
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
            if (ebDecision[Pds::Eb::MON_IDX])  m_mon->post(dgram, ebDecision[Pds::Eb::MON_IDX]);
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
                dmaRetIndexes(m_pool.fd(), m_count, m_indices);
                // std::cout<<"return dma buffers to driver\n";
                m_count = 0;
            }
        }
    }
    event->mask = 0;
}


DrpBase::DrpBase(Parameters& para, ZmqContext& context) :
    pool(para), m_para(para), m_inprocSend(&context, ZMQ_PAIR)
{
    size_t maxSize = sizeof(MyDgram);
    m_tPrms = { /* .ifAddr        = */ { }, // Network interface to use
                /* .port          = */ { }, // Port served to TEBs
                /* .partition     = */ m_para.partition,
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
                /* .partition     = */ m_para.partition,
                /* .id            = */ 0,
                /* .maxEvents     = */ 8,    //mon_buf_cnt,
                /* .maxEvSize     = */ pool.pebble.bufferSize(), //mon_buf_size,
                /* .maxTrSize     = */ 256 * 1024, //mon_trSize,
                /* .verbose       = */ 0 };

    try {
        m_exposer = std::make_unique<prometheus::Exposer>("0.0.0.0:9200", "/metrics", 1);
    }
    catch(const std::runtime_error& e) {
        std::cout<<"Could not start monitoring server!!\n";
        std::cout<<e.what()<<std::endl;
    }

    m_inprocSend.connect("inproc://drp");
}

void DrpBase::shutdown()
{
    if (m_meb) {
        m_meb->shutdown();
    }
}

std::string DrpBase::connect(const json& msg, size_t id)
{
    parseConnectionParams(msg["body"], id);

    m_exporter = std::make_shared<MetricExporter>();
    if (m_exposer) {
        m_exposer->RegisterCollectable(m_exporter);
    }

    // Create all the eb things and do the connections
    m_tebContributor = std::make_unique<Pds::Eb::TebContributor>(m_tPrms, m_exporter);
    int rc = m_tebContributor->connect(m_tPrms);
    if (rc) {
        return std::string{"TebContributor connect failed"};
    }

    if (m_mPrms.addrs.size() != 0) {
        m_meb = std::make_unique<Pds::Eb::MebContributor>(m_mPrms, m_exporter);
        void* poolBase = (void*)pool.pebble[0];
        size_t poolSize = pool.pebble.size();
        rc = m_meb->connect(m_mPrms, poolBase, poolSize);
        if (rc) {
            return std::string{"MebContributor connect failed"};
        }
    }

    m_ebRecv = std::make_unique<EbReceiver>(m_para, m_tPrms, pool, m_inprocSend, m_meb.get(), m_exporter);
    rc = m_ebRecv->connect(m_tPrms);
    if (rc) {
        return std::string{"EbReceiver connect failed"};
    }

    // start eb receiver thread
    m_tebContributor->startup(*m_ebRecv);

    return std::string{};
}

std::string DrpBase::disconnect(const json& msg)
{
    m_tebContributor->stop();
    return std::string{};
}

void DrpBase::parseConnectionParams(const json& body, size_t id)
{
    std::string stringId = std::to_string(id);
    std::cout<<"id  "<<stringId<<std::endl;
    m_tPrms.id = body["drp"][stringId]["drp_id"];
    m_nodeId = body["drp"][stringId]["drp_id"];
    const unsigned numPorts    = Pds::Eb::MAX_DRPS + Pds::Eb::MAX_TEBS + Pds::Eb::MAX_MEBS + Pds::Eb::MAX_MEBS;
    const unsigned tebPortBase = Pds::Eb::TEB_PORT_BASE + numPorts * m_para.partition;
    const unsigned drpPortBase = Pds::Eb::DRP_PORT_BASE + numPorts * m_para.partition;
    const unsigned mebPortBase = Pds::Eb::MEB_PORT_BASE + numPorts * m_para.partition;

    m_tPrms.port = std::to_string(drpPortBase + m_tPrms.id);
    m_mPrms.id = m_tPrms.id;
    m_tPrms.ifAddr = body["drp"][stringId]["connect_info"]["nic_ip"];

    uint64_t builders = 0;
    m_tPrms.addrs.clear();
    m_tPrms.ports.clear();
    for (auto it : body["teb"].items()) {
        unsigned tebId = it.value()["teb_id"];
        std::string address = it.value()["connect_info"]["nic_ip"];
        std::cout << "TEB: " << tebId << "  " << address << '\n';
        builders |= 1ul << tebId;
        m_tPrms.addrs.push_back(address);
        m_tPrms.ports.push_back(std::string(std::to_string(tebPortBase + tebId)));
    }
    m_tPrms.builders = builders;

    m_tPrms.readoutGroup = 1 << unsigned(body["drp"][stringId]["det_info"]["readout"]);
    m_tPrms.contractor = m_tPrms.readoutGroup; // Revisit: Value to come from CfgDb

    m_para.rogMask = 0; // Readout group mask to ignore other partitions' RoGs
    for (auto it : body["drp"].items()) {
        m_para.rogMask |= 1 << unsigned(it.value()["det_info"]["readout"]);
    }

    m_mPrms.addrs.clear();
    m_mPrms.ports.clear();
    m_mPrms.maxEvents = 0;
    if (body.find("meb") != body.end()) {
        for (auto it : body["meb"].items()) {
            unsigned mebId = it.value()["meb_id"];
            std::string address = it.value()["connect_info"]["nic_ip"];
            std::cout << "MEB: " << mebId << "  " << address << '\n';
            m_mPrms.addrs.push_back(address);
            m_mPrms.ports.push_back(std::string(std::to_string(mebPortBase + mebId)));
            unsigned count = it.value()["connect_info"]["buf_count"];
            if (!m_mPrms.maxEvents)  m_mPrms.maxEvents = count;
            if (count != m_mPrms.maxEvents) {
                printf("Error: maxEvents must be the same for all MEBs\n");
            }
        }
    }
}

json DrpBase::connectionInfo()
{
    json info = {{"max_ev_size", m_mPrms.maxEvSize},
                 {"max_tr_size", m_mPrms.maxTrSize}};
    return info;
}

}

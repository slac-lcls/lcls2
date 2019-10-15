#include "BldDetector.hh"

#include <chrono>
#include <iostream>
#include <memory>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include "DataDriver.h"
#include "TimingHeader.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/service/SysLog.hh"
#include <getopt.h>
#include <Python.h>


using json = nlohmann::json;
using logging = Pds::SysLog;

namespace Drp {

static const XtcData::Name::DataType xtype[] = {
    XtcData::Name::UINT8 , // pvBoolean
    XtcData::Name::INT8  , // pvByte
    XtcData::Name::UINT16, // pvShort
    XtcData::Name::INT32 , // pvInt
    XtcData::Name::INT64 , // pvLong
    XtcData::Name::UINT8 , // pvUByte
    XtcData::Name::UINT16, // pvUShort
    XtcData::Name::UINT32, // pvUInt
    XtcData::Name::UINT64, // pvULong
    XtcData::Name::FLOAT , // pvFloat
    XtcData::Name::DOUBLE, // pvDouble
    XtcData::Name::CHARSTR, // pvString
};

unsigned interfaceAddress(const std::string& interface)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    struct ifreq ifr;
    strcpy(ifr.ifr_name, interface.c_str());
    ioctl(fd, SIOCGIFADDR, &ifr);
    close(fd);
    logging::debug("%s", inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr));
    return ntohl(*(unsigned*)&(ifr.ifr_addr.sa_data[2]));
}

XtcData::VarDef BldDescriptor::get(unsigned& payloadSize)
{
    payloadSize = 0;
    XtcData::VarDef vd;
    const pvd::StructureConstPtr& structure = _strct->getStructure();
    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    for (unsigned i=0; i<fields.size(); i++) {
        switch (fields[i]->getType()) {
            case pvd::scalar: {
                const pvd::Scalar* scalar = static_cast<const pvd::Scalar*>(fields[i].get());
                XtcData::Name::DataType type = xtype[scalar->getScalarType()];
                vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type));
                payloadSize += XtcData::Name::get_element_size(type);
                logging::debug("name: %s  %d", names[i].c_str(), type);
                break;
            }

            default: {
                throw std::string("PV type ")+pvd::TypeFunc::name(fields[i]->getType())+
                                  " for field "+names[i]+" not supported";
                break;
            }
        }
    }
    return vd;
}


Bld::Bld(unsigned mcaddr, unsigned port) : m_bufferSize(0), m_position(0), m_first(true),  m_buffer(Bld::MTU)
{
    m_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_sockfd < 0) {
        perror("Open socket");
        throw std::string("Open socket");
    }

    unsigned skbSize = 0x1000000;
    setsockopt(m_sockfd, SOL_SOCKET, SO_RCVBUF, &skbSize, sizeof(skbSize));

    struct sockaddr_in saddr;
    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = htonl(mcaddr);
    saddr.sin_port = htons(port);
    memset(saddr.sin_zero, 0, sizeof(saddr.sin_zero));
    if (bind(m_sockfd, (sockaddr*)&saddr, sizeof(saddr)) < 0) {
        perror("bind");
        throw std::string("bind");
    }

    int y = 1;
    if (setsockopt(m_sockfd, SOL_SOCKET, SO_REUSEADDR, &y, sizeof(y)) == -1) {
        perror("set reuseaddr");
        throw std::string("set reuseaddr");
    }

    unsigned interface = interfaceAddress("eno1");

    ip_mreq ipmreq;
    bzero(&ipmreq, sizeof(ipmreq));
    ipmreq.imr_multiaddr.s_addr = htonl(mcaddr);
    ipmreq.imr_interface.s_addr = htonl(interface);
    if (setsockopt(m_sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                   &ipmreq, sizeof(ipmreq)) == -1) {

        perror("mcast_join");
        throw std::string("mcast_join");
    }
}

Bld::~Bld()
{
    close(m_sockfd);
}

/*
memory layout for bld packet
header:
uint64_t pulseId
uint64_t timeStamp
uint32_t id;
uint8_t payload[]

following events []
uint32_t pulseIdOffset
uint8_t payload[]

*/
uint64_t Bld::next(unsigned payloadSize, uint8_t** payload)
{
    uint64_t pulseId;
    // get new multicast if buffer is empty
    if ((m_position + payloadSize + 4) > m_bufferSize) {
        m_bufferSize = recv(m_sockfd, m_buffer.data(), Bld::MTU, 0);
        pulseId = headerPulseId();
        *payload = &m_buffer[Bld::HeaderSize];
        m_position = Bld::HeaderSize + payloadSize;
    }
    else {
        uint32_t pulseIdOffset = *reinterpret_cast<uint32_t*>(m_buffer.data() + m_position) >> 20;
        pulseId = headerPulseId() + pulseIdOffset;
        *payload = &m_buffer[m_position + 4];
        m_position += 4 + payloadSize;
    }
    return pulseId;
}

class Pgp
{
public:
    Pgp(MemPool& pool, unsigned nodeId, uint32_t envMask) :
        m_pool(pool), m_nodeId(nodeId), m_envMask(envMask), m_available(0), m_current(0)
    {
        uint8_t mask[DMA_MASK_SIZE];
        dmaInitMaskBytes(mask);
        for (unsigned i=0; i<4; i++) {
            dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, 0));
        }
        dmaSetMaskBytes(pool.fd(), mask);
    }

    XtcData::Dgram* next(uint64_t pulseId, uint32_t& evtIndex);
private:
    XtcData::Dgram* handle(Pds::TimingHeader* timingHeader, uint32_t& evtIndex);
    MemPool& m_pool;
    unsigned m_nodeId;
    uint32_t m_envMask;
    int32_t m_available;
    int32_t m_current;
    static const int MAX_RET_CNT_C = 100;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
};

XtcData::Dgram* Pgp::handle(Pds::TimingHeader* timingHeader, uint32_t& evtIndex)
{
    int32_t size = dmaRet[m_current];
    uint32_t index = dmaIndex[m_current];
    uint32_t lane = (dest[m_current] >> 8) & 7;
    if (unsigned(size) > m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d\n", size, m_pool.dmaSize());
        exit(-1);
    }

    const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
    uint32_t evtCounter = data[5] & 0xffffff;
    evtIndex = evtCounter & (m_pool.nbuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[evtIndex];
    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    event->mask |= (1 << lane);

    // make new dgram in the pebble
    XtcData::Dgram* dgram = (XtcData::Dgram*)m_pool.pebble[evtIndex];
    XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
    dgram->xtc.contains = tid;
    dgram->xtc.damage = 0;
    dgram->xtc.extent = sizeof(XtcData::Xtc);

    // fill in dgram header
    dgram->seq = timingHeader->seq;
    dgram->env = timingHeader->env & m_envMask;

    // set the src field for the event builders
    dgram->xtc.src = XtcData::Src(m_nodeId);

    return dgram;
}

XtcData::Dgram* Pgp::next(uint64_t pulseId, uint32_t& evtIndex)
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        auto start = std::chrono::steady_clock::now();
        while (1) {
            m_available = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
            if (m_available > 0) {
                break;
            }

            // wait for a total of 10 ms otherwise timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > 10) {
                // printf("pgp timeout\n");
                return nullptr;
            }
        }
    }

    Pds::TimingHeader* timingHeader = (Pds::TimingHeader*)m_pool.dmaBuffers[dmaIndex[m_current]];

    // return dgram if bld pulseId matches pgp pulseId or if its a transition
    if ((pulseId == timingHeader->seq.pulseId().value()) || (timingHeader->seq.service() != XtcData::TransitionId::L1Accept)) {
        XtcData::Dgram* dgram = handle(timingHeader, evtIndex);
        m_current++;
        return dgram;
    }
    // Missed BLD data so mark event as damaged
    else if (pulseId > timingHeader->seq.pulseId().value()) {
        XtcData::Dgram* dgram = handle(timingHeader, evtIndex);
        dgram->xtc.damage.increase(XtcData::Damage::DroppedContribution);
        m_current++;
        return dgram;
    }

    return nullptr;
}

BldApp::BldApp(Parameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_terminate(false)
{
    logging::info("Ready for transitions");
}

void BldApp::shutdown()
{
    m_exporter.reset();

    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    m_drp.shutdown();
}

json BldApp::connectionInfo()
{
    std::string ip = getNicIp();
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json bufInfo = m_drp.connectionInfo();
    body["connect_info"].update(bufInfo); // Revisit: Should be in det_info
    return body;
}

void BldApp::handleConnect(const nlohmann::json& msg)
{
    json body = json({});
    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect");
        logging::error("%s", errorMsg.c_str());
        body["err_info"] = errorMsg;
    }

    m_pvaAddr = std::make_unique<Pds_Epics::PVBase>("DAQ:LAB2:HPSEX:ADDR");
    m_pvaPort = std::make_unique<Pds_Epics::PVBase>("DAQ:LAB2:HPSEX:PORT");
    m_pvaDescriptor = std::make_unique<BldDescriptor>("DAQ:LAB2:HPSEX:PAYLOAD");

    while(1) {
        if (m_pvaAddr->connected() &&
            m_pvaPort->connected() &&
            m_pvaDescriptor->connected()) {
            break;
        }
        usleep(100000);
    }

    connectPgp(msg, std::to_string(getId()));

    m_unconfigure = false;
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void BldApp::handleDisconnect(const json& msg)
{
    shutdown();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void BldApp::handlePhase1(const json& msg)
{
    logging::debug("handlePhase1 in BldApp");

    json body = json({});
    std::string key = msg["header"]["key"];
    if (key == "configure") {
        if (m_unconfigure) {
            shutdown();
            m_unconfigure = false;
        }

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }

        m_exporter = std::make_shared<MetricExporter>();
        if (m_drp.exposer()) {
            m_drp.exposer()->RegisterCollectable(m_exporter);
        }

        m_terminate.store(false, std::memory_order_release);

        m_workerThread = std::thread{&BldApp::worker, this, m_exporter};
    }
    else if (key == "unconfigure") {
        m_unconfigure = true;
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void BldApp::handleReset(const nlohmann::json& msg)
{
    shutdown();
}

void BldApp::connectPgp(const json& json, const std::string& collectionId)
{
    // FIXME not sure what the size should since for the bld we except to pgp payload
    int length = 4;
    int links = m_para.laneMask;

    int fd = open(m_para.device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para.device.c_str());
    }

    int readoutGroup = json["body"]["drp"][collectionId]["det_info"]["readout"];
    uint32_t v = ((readoutGroup&0xf)<<0) |
                  ((length&0xffffff)<<4) |
                  (links<<28);
    dmaWriteRegister(fd, 0x00a00000, v);
    uint32_t w;
    dmaReadRegister(fd, 0x00a00000, &w);
    logging::info("Configured readout group [%u], length [%u], links [%x]: [%x](%x)",
           readoutGroup, length, links, v, w);
    for (unsigned i=0; i<4; i++) {
        if (links&(1<<i)) {
            // this is the threshold to assert deadtime (high water mark) for every link
            // 0x1f00 corresponds to 0x1f free buffers
            dmaWriteRegister(fd, 0x00800084+32*i, 0x1f00);
        }
    }
    close(fd);
}

void BldApp::worker(std::shared_ptr<MetricExporter> exporter)
{

    unsigned mcaddr = m_pvaAddr->getScalarAs<unsigned>();
    unsigned port = m_pvaPort->getScalarAs<unsigned>();
    logging::debug("addr %u  port %u", mcaddr, port);

    unsigned payloadSize;
    XtcData::VarDef bldDef = m_pvaDescriptor->get(payloadSize);
    logging::debug("payloadSize %u", payloadSize);

    Bld bld(mcaddr, port);
    Pgp pgp(m_drp.pool, m_drp.nodeId(), 0xffff0000 | uint32_t(m_para.rogMask));

    uint64_t nevents = 0L;
    std::map<std::string, std::string> labels{{"partition", std::to_string(m_para.partition)}};
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return nevents;});

    uint64_t nmissed = 0L;
    exporter->add("bld_miss_count", labels, MetricType::Counter,
                  [&](){return nmissed;});

    while (1) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        uint8_t* bldPayload;
        uint64_t pulseId = bld.next(payloadSize, &bldPayload);
        uint32_t index;
        XtcData::Dgram* dgram = pgp.next(pulseId, index);
        if (dgram) {
            if (dgram->xtc.damage.value()) {
                ++nmissed;
                //printf("Missed bld data!!\n");
                //printf("pulseId bld %016lx  | pgp %016lx\n", pulseId, dgram->seq.pulseId().value());
            }
            else {
                XtcData::NamesId namesId(m_drp.nodeId(), 0);
                switch (dgram->seq.service()) {
                    case XtcData::TransitionId::Configure: {
                        logging::info("BLD configure");
                        XtcData::Alg bldAlg("bldAlg", 1, 2, 3);
                        XtcData::Names& bldNames = *new(dgram->xtc) XtcData::Names("bld", bldAlg,
                            "bld", "bld1234", namesId, 0);
                        bldNames.add(dgram->xtc, bldDef);
                        m_nameIndex = XtcData::NameIndex(bldNames);
                        break;
                    }
                    case XtcData::TransitionId::L1Accept: {
                        XtcData::DescribedData desc(dgram->xtc, m_nameIndex, namesId);
                        memcpy(desc.data(), bldPayload, payloadSize);
                        desc.set_data_length(payloadSize);
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
            sentToTeb(*dgram, index);
            nevents++;
        }
    }
    logging::info("Worker thread finished");
}

void BldApp::sentToTeb(XtcData::Dgram& dgram, uint32_t index)
{
    void* buffer = m_drp.tebContributor().allocate(&dgram, (void*)((uintptr_t)index));
    if (buffer) { // else timed out
        PGPEvent* event = &m_drp.pool.pgpEvents[index];
        event->l3InpBuf = buffer;
        XtcData::Dgram* l3InpDg = new(buffer) XtcData::Dgram(dgram);
        if (dgram.seq.isEvent()) {
            if (m_drp.triggerPrimitive()) {// else this DRP doesn't provide input
                m_drp.triggerPrimitive()->event(m_drp.pool, index, dgram.xtc, l3InpDg->xtc); // Produce
            }
        }
        m_drp.tebContributor().process(l3InpDg);
    }
}


} // namespace Drp


int main(int argc, char* argv[])
{
    Drp::Parameters para;
    para.partition = -1;
    para.laneMask = 0x1;
    para.detName = "bld";               // Revisit: Should come from alias?
    para.detSegment = 0;
    para.verbose = 0;
    char *instrument = NULL;
    int c;
    while((c = getopt(argc, argv, "p:o:C:d:u:P:T::v")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.outputDir = optarg;
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'P':
                instrument = optarg;
                break;
            case 'T':
                para.trgDetName = optarg ? optarg : "trigger";
                break;
            case 'v':
                ++para.verbose;
                break;
            default:
                exit(1);
        }
    }

    switch (para.verbose) {
      case 0:  logging::init(instrument, LOG_INFO);     break;
      default: logging::init(instrument, LOG_DEBUG);    break;
    }
    logging::info("logging configured");
    if (!instrument) {
        logging::warning("-P: instrument name is missing");
    }
    // Check required parameters
    if (para.partition == unsigned(-1)) {
        logging::critical("-p: partition is mandatory");
        exit(1);
    }
    if (para.device.empty()) {
        logging::critical("-d: device is mandatory");
        exit(1);
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        exit(1);
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        exit(1);
    }
    para.detName = para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));

    Py_Initialize(); // for use by configuration
    Drp::BldApp app(para);
    app.run();
    app.handleReset(json({}));
    Py_Finalize(); // for use by configuration
}

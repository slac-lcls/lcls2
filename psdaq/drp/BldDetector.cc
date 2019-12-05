#include "BldDetector.hh"

#include <chrono>
#include <iostream>
#include <memory>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include "DataDriver.h"
#include "RunInfoDef.hh"
#include "TimingHeader.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"
#include <getopt.h>
#include <Python.h>


using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Drp {

static const XtcData::Name::DataType xtype[] = {
    XtcData::Name::UINT8 , // pvBoolean
    XtcData::Name::INT8  , // pvByte
    XtcData::Name::INT16,  // pvShort
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

BldPVA::BldPVA(const char* name,
               const char* pvname,
               unsigned    interface) : _name(name), _interface(interface)
{
    std::string sname(pvname);
    _pvaAddr    = std::make_shared<Pds_Epics::PVBase>((sname+":ADDR"   ).c_str());
    _pvaPort    = std::make_shared<Pds_Epics::PVBase>((sname+":PORT"   ).c_str());
    _pvaPayload = std::make_shared<BldDescriptor>((sname+":PAYLOAD").c_str());

    logging::info("BldPVA::BldPVA looking up multicast parameters for %s from %s", name, pvname);
}

BldPVA::~BldPVA()
{
}

  //
  //  LCLS-I Style
  //
BldFactory::BldFactory(const char* name,
                       unsigned    interface) :
  _name       (name),
  _alg        ("bldAlg", 0, 0, 1)
{
    logging::debug("BldFactory::BldFactory %s", name);

    if (strchr(name,':'))
        _name = std::string(strrchr(name,':')+1);

    _pvaPayload = 0;

    unsigned payloadSize = 0;
    unsigned mcaddr = 0;
    unsigned mcport = 12148;
    //
    //  Make static configuration of BLD  :(
    //
    if      (strcmp("ebeam",name)==0) {
        mcaddr = 0xefff1800;
        _alg    = XtcData::Alg((_name+"Alg").c_str(), 0, 7, 1);
        _varDef.NameVec.push_back(XtcData::Name("damageMask"       , XtcData::Name::UINT32));
        _varDef.NameVec.push_back(XtcData::Name("ebeamCharge"      , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamL3Energy"    , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamLTUPosX"     , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamLTUPosY"     , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamLUTAngX"     , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamLTUAngY"     , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamPkCurrBC2"   , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamEnergyBC2"   , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamPkCurrBC1"   , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamEnergyBC1"   , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamUndPosX"     , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamUndPosY"     , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamUndAngX"     , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamUndAngY"     , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamXTCAVAmpl"   , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamXTCAVPhase"  , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamDumpCharge"  , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamPhotonEnergy", XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamLTU250"      , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("ebeamLTU450"      , XtcData::Name::DOUBLE));
        payloadSize = 164;
    }
    else if (strcmp("gasdet",name)==0) {
        mcaddr = 0xefff1802;
        _alg    = XtcData::Alg((_name+"Alg").c_str(), 0, 1, 1);
        _varDef.NameVec.push_back(XtcData::Name("f_11_ENRC"      , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("f_12_ENRC"      , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("f_21_ENRC"      , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("f_22_ENRC"      , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("f_63_ENRC"      , XtcData::Name::DOUBLE));
        _varDef.NameVec.push_back(XtcData::Name("f_64_ENRC"      , XtcData::Name::DOUBLE));
        payloadSize = 24;
    }
    else {
        throw std::string("BLD name ")+name+" not recognized";
    }
    _handler = std::make_shared<Bld>(mcaddr, mcport, interface, Bld::DgramPulseIdPos, Bld::DgramHeaderSize, payloadSize);
}

  //
  //  LCLS-II Style
  //
BldFactory::BldFactory(const BldPVA& pva) :
    _name       (pva._name),
    _alg        ("bldAlg", 0, 0, 1),
    _pvaPayload (pva._pvaPayload)
{
    size_t pos = pva._name.rfind(':');
    if (pos != std::string::npos)
        _name = pva._name.substr(pos+1);

    while(1) {
        if (pva._pvaAddr   ->ready() &&
            pva._pvaPort   ->ready() &&
            pva._pvaPayload->ready())
            break;
        usleep(10000);
    }

    unsigned mcaddr = pva._pvaAddr->getScalarAs<unsigned>();
    unsigned mcport = pva._pvaPort->getScalarAs<unsigned>();

    unsigned payloadSize = 0;
    _varDef = pva._pvaPayload->get(payloadSize);

    if (_name == "hpsex" ||
        _name == "hpscp" ||
        _name == "hpscpb") {
        _alg = XtcData::Alg(_name.c_str(), 0, 0, 1);
        //  validate _varDef against version here
    }
    else {
        throw std::string("BLD name ")+_name+" not recognized";
    }
    _handler = std::make_shared<Bld>(mcaddr, mcport, pva._interface, Bld::PulseIdPos, Bld::HeaderSize, payloadSize);
}

  BldFactory::BldFactory(const BldFactory& o) :
    _name       (o._name),
    _alg        (o._alg),
    _pvaPayload (o._pvaPayload)
{
    logging::error("BldFactory copy ctor called");
}

BldFactory::~BldFactory()
{
  logging::debug("BldFactory::~BldFactory [%s]", _name.c_str());
}

Bld& BldFactory::handler()
{
    return *_handler;
}

XtcData::NameIndex BldFactory::addToXtc  (XtcData::Xtc& xtc,
                                          const XtcData::NamesId& namesId)
{
  XtcData::Names& bldNames = *new(xtc) XtcData::Names(_name.c_str(), _alg,
                                                      _name.c_str(), _name.c_str(), namesId);

  bldNames.add(xtc, _varDef);
  return XtcData::NameIndex(bldNames);
}

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

BldDescriptor::~BldDescriptor()
{
    logging::debug("~BldDescriptor");
}

XtcData::VarDef BldDescriptor::get(unsigned& payloadSize)
{
    payloadSize = 0;
    XtcData::VarDef vd;
    const pvd::StructureConstPtr& structure = _strct->getStructure();
    if (!structure) {
        logging::error("BLD with no payload.  Is FieldMask empty?");
        throw std::string("BLD with no payload.  Is FieldMask empty?");
    }

    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    for (unsigned i=0; i<fields.size(); i++) {
        switch (fields[i]->getType()) {
            case pvd::scalar: {
                const pvd::Scalar* scalar = static_cast<const pvd::Scalar*>(fields[i].get());
                XtcData::Name::DataType type = xtype[scalar->getScalarType()];
                vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type));
                payloadSize += XtcData::Name::get_element_size(type);
                break;
            }

            default: {
                throw std::string("PV type ")+pvd::TypeFunc::name(fields[i]->getType())+
                                  " for field "+names[i]+" not supported";
                break;
            }
        }
    }

    std::string fnames("fields: ");
    for(auto & elem: vd.NameVec)
        fnames += std::string(elem.name()) + "[" + elem.str_type() + "],";
    logging::debug("%s",fnames.c_str());

    return vd;
}

#define HANDLE_ERR(str) {                       \
  perror(str);                                  \
  throw std::string(str); }

Bld::Bld(unsigned mcaddr,
         unsigned port,
         unsigned interface,
         unsigned pulseIdPos,
         unsigned headerSize,
         unsigned payloadSize) :
  m_pulseIdPos(pulseIdPos), m_headerSize(headerSize), m_payloadSize(payloadSize),
  m_bufferSize(0), m_position(0),  m_buffer(Bld::MTU), m_payload(m_buffer.data())
{
    logging::debug("Bld listening for %x.%d with payload size %u",mcaddr,port,payloadSize);

    m_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_sockfd < 0) 
        HANDLE_ERR("Open socket");

    { unsigned skbSize = 0x1000000;
      if (setsockopt(m_sockfd, SOL_SOCKET, SO_RCVBUF, &skbSize, sizeof(skbSize)) == -1) 
          HANDLE_ERR("set so_rcvbuf");
    }

    struct sockaddr_in saddr;
    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = htonl(mcaddr);
    saddr.sin_port = htons(port);
    memset(saddr.sin_zero, 0, sizeof(saddr.sin_zero));
    if (bind(m_sockfd, (sockaddr*)&saddr, sizeof(saddr)) < 0)
        HANDLE_ERR("bind");

    int y = 1;
    if (setsockopt(m_sockfd, SOL_SOCKET, SO_REUSEADDR, &y, sizeof(y)) == -1) 
        HANDLE_ERR("set reuseaddr");

    ip_mreq ipmreq;
    bzero(&ipmreq, sizeof(ipmreq));
    ipmreq.imr_multiaddr.s_addr = htonl(mcaddr);
    ipmreq.imr_interface.s_addr = htonl(interface);
    if (setsockopt(m_sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                   &ipmreq, sizeof(ipmreq)) == -1)
        HANDLE_ERR("mcast join");
}

Bld::Bld(const Bld& o) :
    m_pulseIdPos  (o.m_pulseIdPos),
    m_headerSize  (o.m_headerSize),
    m_payloadSize (o.m_payloadSize),
    m_sockfd      (o.m_sockfd)
{
    logging::error("Bld copy ctor called");
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

uint64_t Bld::next()
{
    uint64_t pulseId(0L);
    // get new multicast if buffer is empty
    if ((m_position + m_payloadSize + 4) > m_bufferSize) {
        m_bufferSize = recv(m_sockfd, m_buffer.data(), Bld::MTU, 0);
        pulseId      = headerPulseId();
        m_payload    = &m_buffer[m_headerSize];
        m_position   = m_headerSize + m_payloadSize;
    }
    else {
        uint32_t pulseIdOffset = *reinterpret_cast<uint32_t*>(m_buffer.data() + m_position) >> 20;
        pulseId     = headerPulseId() + pulseIdOffset;
        m_payload   = &m_buffer[m_position + 4];
        m_position += 4 + m_payloadSize;
    }
    // if (pulseId==0L) {
    //   logging::debug("pulseId is 0");
    // }
    return pulseId;
}

class Pgp
{
public:
    Pgp(MemPool& pool, unsigned nodeId, uint32_t envMask) :
        m_pool(pool), m_nodeId(nodeId), m_envMask(envMask), m_available(0), m_current(0), m_next(0)
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
    uint64_t m_next;
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

static const unsigned _skip_intv = 10000;

XtcData::Dgram* Pgp::next(uint64_t pulseId, uint32_t& evtIndex)
{
    //  Fast forward to _next pulseId
    if (pulseId < m_next)
        return nullptr;

    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        auto start = std::chrono::steady_clock::now();
        while (1) {
            m_available = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
            if (m_available > 0) {
                break;
            }

            //  Timing data should arrive long before BLD - no need to wait
            //            return nullptr;

            // wait for a total of 10 ms otherwise timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > 10) {
                m_next = pulseId + _skip_intv;
                return nullptr;
            }
        }
    }

    Pds::TimingHeader* timingHeader = (Pds::TimingHeader*)m_pool.dmaBuffers[dmaIndex[m_current]];

    // return dgram if bld pulseId matches pgp pulseId or if its a transition
    if ((pulseId == timingHeader->seq.pulseId().value()) || (timingHeader->seq.service() != XtcData::TransitionId::L1Accept)) {
        XtcData::Dgram* dgram = handle(timingHeader, evtIndex);
        m_current++;
        m_next = timingHeader->seq.pulseId().value();
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
    m_drp      (para, context()),
    m_para     (para),
    m_config   (0),
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
    m_namesLookup.clear();
}

json BldApp::connectionInfo()
{
  //
  //  Copied from XpmDetector
  //
    int fd = open(m_para.device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para.device.c_str());
        return json();
    }
    uint32_t reg;
    dmaReadRegister(fd, 0x00a00008, &reg);
    close(fd);

    if (!reg) {
        const char msg[] = "XPM Remote link id register is zero\n";
        logging::error("%s", msg);
        throw msg;
    }
    int x = (reg >> 16) & 0xFF;
    int y = (reg >> 8) & 0xFF;
    int port = reg & 0xFF;
    std::string xpmIp = {"10.0." + std::to_string(x) + '.' + std::to_string(y)};
    json info = {{"xpm_ip", xpmIp}, {"xpm_port", port}};

    std::string ip = getNicIp();
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json bufInfo = m_drp.connectionInfo();
    body["connect_info"].update(bufInfo); // Revisit: Should be in det_info
    body["connect_info"].update(info);
    return body;
}

void BldApp::handleConnect(const nlohmann::json& msg)
{
    json body = json({});
    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in BldApp::handleConnect");
        logging::error("%s", errorMsg.c_str());
        body["err_info"] = errorMsg;
    }

    //  Check for proper command-line parameters
    std::map<std::string,std::string>::iterator it = m_para.kwargs.find("interface");
    if (it == m_para.kwargs.end()) {
        logging::error("Error in BldApp::handleConnect");
        logging::error("No multicast interface specified");
        body["err_info"] = std::string("No multicast interface specified");
    }

    unsigned interface = interfaceAddress(it->second);
    if (!interface) {
        logging::error("Error in BldApp::handleConnect");
        logging::error("Failed to lookup multicast interface %s",it->second.c_str());
        body["err_info"] = std::string("Failed to lookup multicast interface");
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
    json phase1Info{ "" }; 
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});
    std::string key = msg["header"]["key"];

    logging::debug("handlePhase1 for %s in BldApp",key.c_str());

    if (key == "configure") {
        if (m_unconfigure) {
            shutdown();
            m_unconfigure = false;
        }
        else {
            m_exporter = std::make_shared<MetricExporter>();
            if (m_drp.exposer()) {
                m_drp.exposer()->RegisterCollectable(m_exporter);
            }
        }

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }

        m_terminate.store(false, std::memory_order_release);

        m_workerThread = std::thread{&BldApp::worker, this, m_exporter};
    }
    else if (key == "unconfigure") {
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        std::string errorMsg = m_drp.beginrun(phase1Info, m_runInfo); 
        if (!errorMsg.empty()) {
            body["errInfo"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
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
    // FIXME not sure what the size should be since for the bld we expect no pgp payload
    int length = 0;
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
    //
    //  Setup the multicast receivers
    //  
    m_config.erase(m_config.begin(),m_config.end());
    
    unsigned interface = interfaceAddress(m_para.kwargs["interface"]);

    std::vector<std::shared_ptr<BldPVA> > bldPva(0);
    std::string s(m_para.detectorType);
    logging::debug("Parsing %s",s.c_str());
    for(size_t curr = 0, next = 0; next != std::string::npos; curr = next+1) {
        next  = s.find(',',curr+1);
        size_t pvpos = s.find('+',curr+1);
        logging::debug("(%d,%d,%d)",curr,pvpos,next);
        if (next == std::string::npos) {
            if (pvpos != std::string::npos)
                bldPva.push_back(std::make_shared<BldPVA>(s.substr(curr,pvpos-curr).c_str(),
                                                          s.substr(pvpos+1,-1).c_str(),
                                                          interface));
            else
                m_config.push_back(std::make_shared<BldFactory>(s.substr(curr,next).c_str(),
                                                                interface));
        }
        else if (pvpos > curr && pvpos < next)
            bldPva.push_back(std::make_shared<BldPVA>(s.substr(curr,pvpos-curr).c_str(),
                                                      s.substr(pvpos+1,next-pvpos-1).c_str(),
                                                      interface));
        else
            m_config.push_back(std::make_shared<BldFactory>(s.substr(curr,next-curr).c_str(),
                                                            interface));
    }

    for(unsigned i=0; i<bldPva.size(); i++)
        m_config.push_back(std::make_shared<BldFactory>(*bldPva[i].get()));

    //    std::vector<XtcData::NameIndex> nameIndex(m_config.size());

    Pgp pgp(m_drp.pool, m_drp.nodeId(), 0xffff0000 | uint32_t(m_para.rogMask));

    uint64_t nevents = 0L;
    std::map<std::string, std::string> labels{{"partition", std::to_string(m_para.partition)}};
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return nevents;});

    uint64_t nmissed = 0L;
    exporter->add("bld_miss_count", labels, MetricType::Counter,
                  [&](){return nmissed;});

    uint64_t nextId = -1ULL;
    std::vector<uint64_t> pulseId(m_config.size());
    for(unsigned i=0; i<m_config.size(); i++) {
        pulseId[i] = m_config[i]->handler().next();
        if (pulseId[i] < nextId)
            nextId = pulseId[i];
        logging::info("BldApp::worker Initial pulseId[%d] 0x%" PRIx64, i, pulseId[i]);
    }

    bool lMissing = false;

    while (1) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        uint32_t index;
        XtcData::Dgram* dgram = pgp.next(nextId, index);
        bool lHold(false);
        if (dgram) {
            if (dgram->xtc.damage.value()) {
                ++nmissed;
                if (dgram->seq.pulseId().value() < nextId)
                    lHold=true;
                if (!lMissing) {
                    lMissing = true;
                    logging::debug("Missed next bld: pgp %016lx  bld %016lx",
                                   dgram->seq.pulseId().value(), nextId);
                }
            }
            else {
                switch (dgram->seq.service()) {
                    case XtcData::TransitionId::Configure: {
                        logging::info("BLD configure");
                        for(unsigned i=0; i<m_config.size(); i++) {
                            XtcData::NamesId namesId(m_drp.nodeId(), i);
                            m_namesLookup[namesId] = m_config[i]->addToXtc(dgram->xtc, namesId);
                        }
                        m_drp.runInfoSupport(dgram->xtc, m_namesLookup);

                        if (dgram->xtc.extent > m_drp.pool.bufferSize()) {
                            logging::critical("Transition: buffer size (%d) too small for requested extent (%d)", m_drp.pool.bufferSize(), dgram->xtc.extent);
                            exit(-1);
                        }

                        lHold=true;
                        break;
                    }
                    case XtcData::TransitionId::BeginRun: {
                        if (m_runInfo.runNumber > 0) {
                            m_drp.runInfoData(dgram->xtc, m_namesLookup, m_runInfo);
                        }
                        break;
                    }
                    case XtcData::TransitionId::L1Accept: {
                        bool lMissed = false;
                        for(unsigned i=0; i<m_config.size(); i++) {
                            if (pulseId[i] == nextId) {
                                XtcData::NamesId namesId(m_drp.nodeId(), i);
                                const Bld& bld = m_config[i]->handler();
                                XtcData::DescribedData desc(dgram->xtc, m_namesLookup, namesId);
                                memcpy(desc.data(), bld.payload(), bld.payloadSize());
                                desc.set_data_length(bld.payloadSize());
                            }
                            else {
                              lMissed = true;
                              if (!lMissing)
                                logging::debug("Missed bld[%u]: pgp %016lx  bld %016lx",
                                               i, nextId, pulseId[i]);
                            }
                        }
                        if (lMissed) {
                            lMissing = true;
                            nmissed++;
                            dgram->xtc.damage.increase(XtcData::Damage::DroppedContribution);
                        }
                        else if (lMissing) {
                            lMissing = false;
                            logging::debug("Missing ends: pgp %016lx", nextId);
                        }
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

        if (!lHold) {
            nextId++;
            for(unsigned i=0; i<m_config.size(); i++) {
                if (pulseId[i] < nextId)
                  pulseId[i] = m_config[i]->handler().next();
            }


            nextId = -1ULL;
            for(unsigned i=0; i<m_config.size(); i++) {
                if (pulseId[i] < nextId)
                    nextId = pulseId[i];
            }
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


static void get_kwargs(Drp::Parameters& para, const std::string& kwargs_str) {
    std::istringstream ss(kwargs_str);
    std::string kwarg;
    std::string::size_type pos = 0;
    while (getline(ss, kwarg, ',')) {
        pos = kwarg.find("=", pos);
        if (!pos) {
            throw "drp.cc error: keyword argument with no equal sign: "+kwargs_str;
        }
        std::string key = kwarg.substr(0,pos);
        std::string value = kwarg.substr(pos+1,kwarg.length());
        //cout << kwarg << " " << key << " " << value << endl;
        para.kwargs[key] = value;
    }
}


int main(int argc, char* argv[])
{
    Drp::Parameters para;
    para.partition = -1;
    para.laneMask = 0x1;
    para.detName = "bld";               // Revisit: Should come from alias?
    para.detSegment = 0;
    para.verbose = 0;
    std::string kwargs_str;
    char *instrument = NULL;
    int c;
    while((c = getopt(argc, argv, "l:p:o:C:b:d:D:u:P:T::k:v")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'l':
                para.laneMask = strtoul(optarg,NULL,0);
                break;
            case 'o':
                para.outputDir = optarg;
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'b':
                para.detName = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'D':
                para.detectorType = optarg;
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
            case 'k':
                kwargs_str = std::string(optarg);
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
    //    para.detName = para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));
    get_kwargs(para, kwargs_str);

    Py_Initialize(); // for use by configuration
    Drp::BldApp app(para);
    app.run();
    app.handleReset(json({}));
    Py_Finalize(); // for use by configuration
}

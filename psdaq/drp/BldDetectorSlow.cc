#define __STDC_FORMAT_MACROS 1

#include "BldDetector.hh"

#include <bitset>
#include <chrono>
#include <iostream>
#include <memory>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <net/if.h>
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "TebReceiver.hh"
#include "RunInfoDef.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"
#include <getopt.h>
#include <Python.h>
#include <inttypes.h>
#include <poll.h>

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using namespace XtcData;
using namespace Pds;
using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

namespace Drp {

static const Name::DataType xtype[] = {
    Name::UINT8 , // pvBoolean
    Name::INT8  , // pvByte
    Name::INT16,  // pvShort
    Name::INT32 , // pvInt
    Name::INT64 , // pvLong
    Name::UINT8 , // pvUByte
    Name::UINT16, // pvUShort
    Name::UINT32, // pvUInt
    Name::UINT64, // pvULong
    Name::FLOAT , // pvFloat
    Name::DOUBLE, // pvDouble
    Name::CHARSTR, // pvString
};

BldPVA::BldPVA(std::string det,
               unsigned    interface) : _interface(interface)
{
    //
    //  Parse '+' separated list of detName, detType, detId
    //
    size_t p1 = det.find('+',0);
    if (p1 == std::string::npos) {
    }
    size_t p2 = det.find('+',p1+1);
    if (p2 == std::string::npos) {
    }

    _detName = det.substr(   0,     p1).c_str();
    _detType = det.substr(p1+1,p2-p1-1).c_str();
    _detId   = det.substr(p2+1).c_str();

    std::string sname(_detId);
    _pvaAddr    = std::make_shared<Pds_Epics::PVBase>((sname+":ADDR"   ).c_str());
    _pvaPort    = std::make_shared<Pds_Epics::PVBase>((sname+":PORT"   ).c_str());
    _pvaPayload = std::make_shared<BldDescriptor>    ((sname+":PAYLOAD").c_str());

    logging::info("BldPVA::BldPVA looking up multicast parameters for %s/%s from %s",
                  _detName.c_str(), _detType.c_str(), _detId.c_str());
}

BldPVA::~BldPVA()
{
}

  //
  //  LCLS-I Style
  //
BldFactory::BldFactory(const char* name,
                       unsigned    interface) :
  _alg        ("raw", 2, 0, 0)
{
    logging::debug("BldFactory::BldFactory %s", name);

    if (strchr(name,':'))
        name = strrchr(name,':')+1;

    _detName = std::string(name);
    _detType = std::string(name);
    _detId   = std::string(name);

    _pvaPayload = 0;

    unsigned payloadSize = 0;
    unsigned mcaddr = 0;
    unsigned mcport = 10148; // 12148, eventually
    uint64_t tscorr = 0x259e9d80ULL << 32;
    //
    //  Make static configuration of BLD  :(
    //
    if      (strncmp("ebeam",name,5)==0) {
        if (name[5]=='h') {
            mcaddr = 0xefff1800;
        }
        else {
            mcaddr = 0xefff1900;
        }
        tscorr = 0;
        _alg    = Alg("raw", 2, 0, 0);
        _varDef.NameVec.push_back(Name("damageMask"       , Name::UINT32));
        _varDef.NameVec.push_back(Name("ebeamCharge"      , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamL3Energy"    , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamLTUPosX"     , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamLTUPosY"     , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamLUTAngX"     , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamLTUAngY"     , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamPkCurrBC2"   , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamEnergyBC2"   , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamPkCurrBC1"   , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamEnergyBC1"   , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamUndPosX"     , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamUndPosY"     , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamUndAngX"     , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamUndAngY"     , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamXTCAVAmpl"   , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamXTCAVPhase"  , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamDumpCharge"  , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamPhotonEnergy", Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamLTU250"      , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ebeamLTU450"      , Name::DOUBLE));
        payloadSize = 164;
    }
    else if (strncmp("pcav",name,4)==0) {
        if (name[4]=='h') {
            mcaddr = 0xefff1801;
        }
        else {
            mcaddr = 0xefff1901;
        }
        _alg    = Alg("raw", 2, 0, 0);
        _varDef.NameVec.push_back(Name("fitTime1"      , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("fitTime2"      , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("charge1"       , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("charge2"       , Name::DOUBLE));
        payloadSize = 32;
    }
    else if (strncmp("gmd",name,3)==0) {
        mcaddr = 0xefff1902;
        _alg    = Alg("raw", 2, 1, 0);
        _varDef.NameVec.push_back(Name("energy"      , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("xpos"        , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ypos"        , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("avgIntensity", Name::DOUBLE));
        _varDef.NameVec.push_back(Name("rmsElectronSum", Name::INT64));
        _varDef.NameVec.push_back(Name("electron1BkgNoiseAvg", Name::INT16));
        _varDef.NameVec.push_back(Name("electron2BkgNoiseAvg", Name::INT16));
        payloadSize = 44;
    }
    else if (strcmp("xgmd",name)==0) {
        mcaddr = 0xefff1903;
        _alg    = Alg("raw", 2, 1, 0);
        _varDef.NameVec.push_back(Name("energy"      , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("xpos"        , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("ypos"        , Name::DOUBLE));
        _varDef.NameVec.push_back(Name("avgIntensity", Name::DOUBLE));
        _varDef.NameVec.push_back(Name("rmsElectronSum", Name::INT64));
        _varDef.NameVec.push_back(Name("electron1BkgNoiseAvg", Name::INT16));
        _varDef.NameVec.push_back(Name("electron2BkgNoiseAvg", Name::INT16));
        payloadSize = 44;
    }
    else {
        throw std::string("BLD name ")+name+" not recognized";
    }
    _handler = std::make_shared<Bld>(mcaddr, mcport, interface,
                                     Bld::DgramTimestampPos, Bld::DgramPulseIdPos,
                                     Bld::DgramHeaderSize, payloadSize,
                                     tscorr);
}

  //
  //  LCLS-II Style
  //
BldFactory::BldFactory(const BldPVA& pva) :
    _detName    (pva._detName),
    _detType    (pva._detType),
    _detId      (pva._detId),
    _alg        ("raw", 2, 0, 0),
    _pvaPayload (pva._pvaPayload)
{
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

    if (_detType == "hpsex" ||
        _detType == "hpscp" ||
        _detType == "hpscpb") {
        _alg = Alg("raw", 2, 0, 0);
        //  validate _varDef against version here
    }
    else {
        throw std::string("BLD type ")+_detType+" not recognized";
    }
    _handler = std::make_shared<Bld>(mcaddr, mcport, pva._interface,
                                     Bld::TimestampPos, Bld::PulseIdPos,
                                     Bld::HeaderSize, payloadSize);
}

  BldFactory::BldFactory(const BldFactory& o) :
    _detName    (o._detName),
    _detType    (o._detType),
    _detId      (o._detId),
    _alg        (o._alg),
    _pvaPayload (o._pvaPayload)
{
    logging::error("BldFactory copy ctor called");
}

BldFactory::~BldFactory()
{
}

Bld& BldFactory::handler()
{
    return *_handler;
}

NameIndex BldFactory::addToXtc  (Xtc& xtc,
                                          const void* bufEnd,
                                          const NamesId& namesId)
{
    Names& bldNames = *new(xtc, bufEnd) Names(bufEnd,
                                                                _detName.c_str(), _alg,
                                                                _detType.c_str(), _detId.c_str(), namesId);

    bldNames.add(xtc, bufEnd, _varDef);
    return NameIndex(bldNames);
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

VarDef BldDescriptor::get(unsigned& payloadSize)
{
    payloadSize = 0;
    VarDef vd;
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
                Name::DataType type = xtype[scalar->getScalarType()];
                vd.NameVec.push_back(Name(names[i].c_str(), type));
                payloadSize += Name::get_element_size(type);
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
         unsigned timestampPos,
         unsigned pulseIdPos,
         unsigned headerSize,
         unsigned payloadSize,
         uint64_t timestampCorr) :
  m_timestampPos(timestampPos), m_pulseIdPos(pulseIdPos),
  m_headerSize(headerSize), m_payloadSize(payloadSize),
  m_bufferSize(0), m_position(0),  m_buffer(Bld::MTU), m_payload(m_buffer.data()),
  m_timestampCorr(timestampCorr), m_pulseId(0), m_pulseIdJump(0)
{
    logging::debug("Bld listening for %x.%d with payload size %u",mcaddr,port,payloadSize);

    m_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_sockfd < 0)
        HANDLE_ERR("Open socket");

    //  Yes, we do bump into full buffers.  Bigger or small buffers seem to be worse.
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
    m_timestampPos(o.m_timestampPos),
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

//  Read ahead and clear events older than ts (approximate)
void     Bld::clear(uint64_t ts)
{
    uint64_t timestamp(0L);
    uint64_t pulseId  (0L);
    while(1) {
        // get new multicast if buffer is empty
        if ((m_position + m_payloadSize + 4) > m_bufferSize) {
            ssize_t bytes = recv(m_sockfd, m_buffer.data(), Bld::MTU, MSG_DONTWAIT);
            if (bytes <= 0)
                break;
            m_bufferSize = bytes;
            timestamp    = headerTimestamp();
            if (timestamp >= ts) {
                m_position = 0;
                break;
            }
            pulseId      = headerPulseId  ();
            m_payload    = &m_buffer[m_headerSize];
            m_position   = m_headerSize + m_payloadSize;
        }
        else if (m_position==0) {
            timestamp    = headerTimestamp();
            if (timestamp >= ts)
                break;
            pulseId      = headerPulseId  ();
            m_payload    = &m_buffer[m_headerSize];
            m_position   = m_headerSize + m_payloadSize;
        }
        else {
            uint32_t timestampOffset = *reinterpret_cast<uint32_t*>(m_buffer.data() + m_position)&0xfffff;
            timestamp   = headerTimestamp() + timestampOffset;
            if (timestamp >= ts)
                break;
            uint32_t pulseIdOffset   = (*reinterpret_cast<uint32_t*>(m_buffer.data() + m_position)>>20)&0xfff;
            pulseId     = headerPulseId  () + pulseIdOffset;
            m_payload   = &m_buffer[m_position + 4];
            m_position += 4 + m_payloadSize;
        }

        unsigned jump = pulseId - m_pulseId;
        m_pulseId = pulseId;
        if (jump != m_pulseIdJump) {
            m_pulseIdJump = jump;
            logging::warning("BLD pulseId jump %u",jump);
        }
    }
}

//  Advance to the next event
uint64_t Bld::next()
{
    uint64_t timestamp(0L);
    uint64_t pulseId  (0L);
    // get new multicast if buffer is empty
    if ((m_position + m_payloadSize + 4) > m_bufferSize) {
        ssize_t bytes = recv(m_sockfd, m_buffer.data(), Bld::MTU, 0);
        m_bufferSize = bytes;
        timestamp    = headerTimestamp();
        pulseId      = headerPulseId  ();
        m_payload    = &m_buffer[m_headerSize];
        m_position   = m_headerSize + m_payloadSize;
    }
    else if (m_position==0) {
        timestamp    = headerTimestamp();
        pulseId      = headerPulseId  ();
        m_payload    = &m_buffer[m_headerSize];
        m_position   = m_headerSize + m_payloadSize;
    }
    else {
        uint32_t timestampOffset = *reinterpret_cast<uint32_t*>(m_buffer.data() + m_position)&0xfffff;
        timestamp   = headerTimestamp() + timestampOffset;
        uint32_t pulseIdOffset   = (*reinterpret_cast<uint32_t*>(m_buffer.data() + m_position)>>20)&0xfff;
        pulseId     = headerPulseId  () + pulseIdOffset;
        m_payload   = &m_buffer[m_position + 4];
        m_position += 4 + m_payloadSize;
    }

    unsigned jump = pulseId - m_pulseId;
    m_pulseId = pulseId;
    if (jump != m_pulseIdJump) {
        m_pulseIdJump = jump;
        logging::warning("BLD pulseId jump %u",jump);
    }

    return timestamp;
}


class BldDetector : public XpmDetector
{
public:
    BldDetector(Parameters& para, MemPoolCpu& pool) : XpmDetector(&para, &pool) { virtChan = 0; }
    void event(Dgram& dgram, const void* bufEnd, PGPEvent* event) override {}
};


Pgp::Pgp(Parameters& para, DrpBase& drp, Detector* det) :
    PgpReader(para, drp.pool, MAX_RET_CNT_C, 32),
    m_para(para), m_drp(drp), m_det(det),
    m_config(0), m_terminate(false), m_running(false),
    m_available(0), m_current(0), m_next(0), m_nDmaRet(0)
{
    if (drp.pool.setMaskBytes(para.laneMask, det->virtChan)) {
        logging::error("Failed to allocate lane/vc");
    }
}

EbDgram* Pgp::_handle(uint32_t& evtIndex)
{
    const TimingHeader* timingHeader = handle(m_det, m_current);
    if (!timingHeader)  return nullptr;

    uint32_t pgpIndex = timingHeader->evtCounter & (m_pool.nDmaBuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[pgpIndex];

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    evtIndex = event->pebbleIndex;
    Src src = m_det->nodeId;
    EbDgram* dgram = new(m_pool.pebble[evtIndex]) EbDgram(*timingHeader, src, m_para.rogMask);

    // Collect indices of DMA buffers that can be recycled and reset event
    freeDma(event);

    return dgram;
}

EbDgram* Pgp::next(uint32_t& evtIndex)
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        m_available = read();
        m_nDmaRet = m_available;
        if (m_available == 0)
            return nullptr;
    }

    EbDgram* dgram = _handle(evtIndex);
    m_current++;
    return dgram;
}

void Pgp::shutdown()
{
    m_terminate.store(true, std::memory_order_release);
    m_det->namesLookup().clear();   // erase all elements
}

int Pgp::_setupMetrics(const std::shared_ptr<MetricExporter> exporter)
{
    // setup monitoring
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"detseg", std::to_string(m_para.detSegment)},
                                              {"alias", m_para.alias}};
    m_nevents = 0L;
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return m_nevents;});
    m_nmissed = 0L;
    exporter->add("bld_miss_count", labels, MetricType::Counter,
                  [&](){return m_nmissed;});

    m_nDmaRet = 0L;
    exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                  [&](){return m_nDmaRet;});
    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return dmaBytes();});
    exporter->add("drp_dma_size", labels, MetricType::Gauge,
                  [&](){return dmaSize();});
    exporter->add("drp_th_latency", labels, MetricType::Gauge,
                  [&](){return latency();});
    exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                  [&](){return nDmaErrors();});
    exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                  [&](){return nNoComRoG();});
    exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                  [&](){return nMissingRoGs();});
    exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                  [&](){return nTmgHdrError();});
    exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                  [&](){return nPgpJumps();});
    exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                  [&](){return nNoTrDgrams();});

    exporter->add("drp_num_pgp_in_user", labels, MetricType::Gauge,
                  [&](){return nPgpInUser();});
    exporter->add("drp_num_pgp_in_hw", labels, MetricType::Gauge,
                  [&](){return nPgpInHw();});
    exporter->add("drp_num_pgp_in_prehw", labels, MetricType::Gauge,
                  [&](){return nPgpInPreHw();});
    exporter->add("drp_num_pgp_in_rx", labels, MetricType::Gauge,
                  [&](){return nPgpInRx();});

    return 0;
}

void Pgp::worker(std::shared_ptr<MetricExporter> exporter)
{
    logging::info("Worker thread is starting with process ID %lu", syscall(SYS_gettid));
    if (prctl(PR_SET_NAME, "drp_bld/Worker", 0, 0, 0) == -1) {
        perror("prctl");
    }

    // Reset counters to avoid 'jumping' errors on reconfigures
    m_pool.resetCounters();
    resetEventCounter();

    // Set up monitoring
    if (exporter) {
        if (_setupMetrics(exporter))  return;
    }

    //
    //  Setup the multicast receivers
    //
    m_config.erase(m_config.begin(), m_config.end());

    unsigned interface = interfaceAddress(m_para.kwargs["interface"]);

    //
    //  Cache the BLD types that require lookup
    //
    std::vector<std::shared_ptr<BldPVA> > bldPva(0);

    std::string s(m_para.detType);
    logging::debug("Parsing %s",s.c_str());
    for(size_t curr = 0, next = 0; next != std::string::npos; curr = next+1) {
        if (s==".") break;
        next  = s.find(',',curr+1);
        size_t pvpos = s.find('+',curr+1);
        logging::debug("(%d,%d,%d)",curr,pvpos,next);
        if (next == std::string::npos) {
            if (pvpos != std::string::npos)
                bldPva.push_back(std::make_shared<BldPVA>(s.substr(curr,next),
                                                          interface));
            else
                m_config.push_back(std::make_shared<BldFactory>(s.substr(curr,next).c_str(),
                                                                interface));
        }
        else if (pvpos > curr && pvpos < next)
            bldPva.push_back(std::make_shared<BldPVA>(s.substr(curr,next-curr),
                                                      interface));
        else
            m_config.push_back(std::make_shared<BldFactory>(s.substr(curr,next-curr).c_str(),
                                                            interface));
    }

    for(unsigned i=0; i<bldPva.size(); i++)
        m_config.push_back(std::make_shared<BldFactory>(*bldPva[i].get()));

    //  Event builder variables
    unsigned index;
    EbDgram* dgram = 0;
    uint64_t timestamp[m_config.size()];
    memset(timestamp,0,sizeof(timestamp));
    bool lMissing = false;
    NamesLookup& namesLookup = m_det->namesLookup();

    //  Poll
    // this was 4ms, but EBeam bld timed out in rix intermittently,
    // increased it to 50ms, but then we get deadtime running bld
    // with no eventcode 136 @120Hz.  120Hz corresponds to 8ms, so try 7ms.
    unsigned tmo = 7; // milliseconds
    {
        std::map<std::string,std::string>::iterator it = m_para.kwargs.find("timeout");
        if (it != m_para.kwargs.end())
            tmo = strtoul(it->second.c_str(),NULL,0);
    }

    //bool lRunning = false;
    bool lDoPoll = true;    // make the system call
    unsigned skipPoll = 0;  // bit mask of contributors with data waiting

    unsigned nfds = m_config.size()+1;
    pollfd pfd[nfds];
    pfd[0].fd = m_drp.pool.fd();
    pfd[0].events = POLLIN;
    for(unsigned i=0; i<m_config.size(); i++) {
        pfd[i+1].fd = m_config[i]->handler().fd();
        pfd[i+1].events = POLL_IN;
    }
    m_terminate.store(false, std::memory_order_release);

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        timespec tto;
        clock_gettime(CLOCK_REALTIME,&tto);

        int rval;
        bool ltmo=false;
        if (lDoPoll) {
            rval = poll(pfd, nfds, tmo);
            ltmo = rval==0;
        }
        else {
            rval = 0;
            for(unsigned i=0; i<nfds; i++)
                pfd[i].revents = 0;
        }

        if (dgram==0 && pfd[0].events==0 && (skipPoll&1)==0)
            logging::critical("not waiting for dgram");

        if (rval < 0) {  // error
        }
        else {
            lDoPoll = false;

            // handle pgp
            if ((skipPoll&(1<<0)) || pfd[0].revents == POLLIN) {
                dgram = next(index);
                if (dgram) {
                    pfd[0].events = 0;
                    skipPoll &= ~(1<<0);
                }
            }

            // handle bld
            for(unsigned i=0; i<m_config.size(); i++) {
                if (dgram)
                    m_config[i]->handler().clear(dgram->time.value());
                if ((skipPoll&(2<<i)) || pfd[i+1].revents == POLLIN) {
                    timestamp[i] = m_config[i]->handler().next();
                    pfd[i+1].events = 0;
                    skipPoll &= ~(2<<i);
                }
            }

            if (dgram) {
                bool lready = true;
                uint64_t ts = dgram->time.value();
                // handle bld
                for(unsigned i=0; i<m_config.size(); i++) {
                    if (timestamp[i] < ts) {
                        if (m_config[i]->handler().ready())
                            skipPoll |= (2<<i);
                        else {
                            lDoPoll = true;
                            pfd[i+1].events = POLLIN;
                        }
                        lready = false;
                    }
                }
                //  Accept non-L1 transitions
                if (dgram->service() != TransitionId::L1Accept) {

                    // Find the transition dgram in the pool and initialize its header
                    EbDgram* trDgram = m_drp.pool.transitionDgrams[index];
                    const void*   bufEnd  = (char*)trDgram + m_para.maxTrSize;
                    if (!trDgram)  continue; // Can happen during shutdown
                    memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    Xtc& trXtc = m_det->transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                    if (dgram->service() == TransitionId::Configure) {
                        logging::info("BLD configure");
                        // Revisit: This is intended to be done by BldDetector::configure()
                        for(unsigned i=0; i<m_config.size(); i++) {
                            NamesId namesId(m_det.nodeId, BldNamesIndex + i);
                            namesLookup[namesId] = m_config[i]->addToXtc(trDgram->xtc, bufEnd, namesId);
                        }
                    }
                    //else if (dgram->service() == TransitionId::Enable)
                    //    lRunning = true;
                    //else if (dgram->service() == TransitionId::Disable)
                    //    lRunning = false;

                    _sendToTeb(*dgram, index);
                    m_nevents++;
                    if (_ready())
                        skipPoll |= (1<<0);
                    else {
                        pfd[0].events = POLLIN;
                        lDoPoll = true;
                    }
                    dgram = 0;
                }
                //  Accept L1 transitions
                else if (lready or ltmo) {
                    const void* bufEnd = (char*)dgram + m_drp.pool.pebble.bufferSize();
                    bool lMissed = false;
                    for(unsigned i=0; i<m_config.size(); i++) {
                        if (timestamp[i] == ts) {
                            NamesId namesId(m_det.nodeId, BldNamesIndex + i);
                            const Bld& bld = m_config[i]->handler();
                            DescribedData desc(dgram->xtc, bufEnd, namesLookup, namesId);
                            memcpy(desc.data(), bld.payload(), bld.payloadSize());
                            desc.set_data_length(bld.payloadSize());

                            if (bld.ready())
                                skipPoll |= (2<<i);
                            else {
                                lDoPoll = true;
                                pfd[i+1].events = POLLIN;
                            }
                        }
                        else {
                            lMissed = true;
                            if (!lMissing)
                                logging::warning("Missed bld[%u]: pgp %016lx  bld %016lx  pid %016llx",
                                                 i, ts, timestamp[i], dgram->pulseId());
                        }
                    }
                    if (lMissed) {
                        lMissing = true;
                        dgram->xtc.damage.increase(Damage::DroppedContribution);
                        m_nmissed++;
                    }
                    else {
                        if (lMissing)
                            logging::warning("Found bld: %016lx  %016llx",ts, dgram->pulseId());
                        lMissing = false;
                    }

                    _sendToTeb(*dgram, index);
                    m_nevents++;
                    if (_ready())
                        skipPoll |= (1<<0);
                    else {
                        pfd[0].events = POLLIN;
                        lDoPoll = true;
                    }
                    dgram = 0;
                }
            }
            else {  // dgram==0
                //  dgram contributor
                if (_ready())
                    skipPoll |= (1<<0);
                else {
                    lDoPoll = true;
                    pfd[0].events = POLLIN;
                }
                if (ltmo) {
                    // timedout: free up some network buffers
                    uint64_t tt = (tto.tv_sec - POSIX_TIME_AT_EPICS_EPOCH);
                    uint32_t tb = 15000000;  // allow 15 ms for transit+NTP (+2.7ms from xtpg timestamp casting)
                    if (tto.tv_nsec > tb) {
                        tt <<= 32;
                        tt  += tto.tv_nsec-tb;
                    }
                    else {
                        tt--;
                        tt <<= 32;
                        tt  += 1000000000+tto.tv_nsec-tb;
                    }
                    for(unsigned i=0; i<m_config.size(); i++) {
                        m_config[i]->handler().clear(tt);
                        if (m_config[i]->handler().ready())
                            skipPoll |= (2<<i);
                        else {
                            lDoPoll = true;
                            pfd[i+1].events = POLLIN;
                        }
                    }
                }
                // Time out batches for the TEB
                m_drp.tebContributor().timeout();
            }
        }
    }

    // Flush the DMA buffers
    flush();

    logging::info("Worker thread finished");
}

void Pgp::_sendToTeb(EbDgram& dgram, uint32_t index)
{
    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = (dgram.service() == TransitionId::L1Accept)
                         ? m_drp.pool.pebble.bufferSize()
                         : m_para.maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", TransitionId::name(dgram.service()), size, maxSize);
        throw "Dgram overflowed buffer";
    }

    auto l3InpBuf = m_drp.tebContributor().fetch(index);
    EbDgram* l3InpDg = new(l3InpBuf) EbDgram(dgram);
    if (l3InpDg->isEvent()) {
        auto triggerPrimitive = m_drp.triggerPrimitive();
        if (triggerPrimitive) { // else this DRP doesn't provide input
            const void* bufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
            triggerPrimitive->event(m_drp.pool, index, dgram.xtc, l3InpDg->xtc, bufEnd); // Produce
        }
    }
    m_drp.tebContributor().process(l3InpDg);
}


BldDrp::BldDrp(Parameters& para, MemPoolCpu& pool, Detector& det, ZmqContext& context) :
    DrpBase(para, pool, det, context),
    m_pgp  (para, *this, &det)
{
    // Set the TebReceiver we will use in the base class
    setTebReceiver(std::make_unique<TebReceiver>(para, *this));
}

std::string BldDrp::configure(const json& msg)
{
    std::string errorMsg = DrpBase::configure(msg);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    if (exposer()) {
        m_exporter = std::make_shared<MetricExporter>();
        exposer()->RegisterCollectable(m_exporter);
    }

    m_workerThread = std::thread{&Pgp::worker, &m_pgp, m_exporter};

    return std::string();
}

unsigned BldDrp::unconfigure()
{
    DrpBase::unconfigure(); // TebContributor must be shut down before the worker

    m_pgp.shutdown();
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }

    if (m_exporter)  m_exporter.reset();

    return 0;
}


BldApp::BldApp(Parameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_para       (para),
    m_pool       (para),
    m_unconfigure(false)
{
    Py_Initialize();                    // for use by configuration

    m_det = std::make_unique<BldDetector>(m_para, m_pool);
    m_drp = std::make_unique<BldDrp>(m_para, m_pool, *m_det, context());

    logging::info("Ready for transitions");
}

BldApp::~BldApp()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    Py_Finalize();                      // for use by configuration
}

void BldApp::_disconnect()
{
    m_drp->disconnect();
    m_det->shutdown();
}

void BldApp::_unconfigure()
{
    m_drp->pool.shutdown();  // Release Tr buffer pool
    m_drp->unconfigure();
    m_unconfigure = false;
}

json BldApp::connectionInfo(const json& msg)
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = static_cast<Detector&>(*m_det).connectionInfo(msg);
    body["connect_info"].update(info);
    json bufInfo = m_drp->connectionInfo(ip);
    body["connect_info"].update(bufInfo);
    return body;
}

void BldApp::connectionShutdown()
{
    static_cast<Detector&>(*m_det).connectionShutdown();
    m_drp->shutdown();
}

void BldApp::_error(const std::string& which, const json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void BldApp::handleConnect(const json& msg)
{
    std::string errorMsg = m_drp->connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in BldApp::handleConnect");
        logging::error("%s", errorMsg.c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    //  Check for proper command-line parameters
    std::map<std::string,std::string>::iterator it = m_para.kwargs.find("interface");
    if (it == m_para.kwargs.end()) {
        logging::error("Error in BldApp::handleConnect");
        logging::error("No multicast interface specified");
        _error("connect", msg, std::string("No multicast interface specified"));
        return;
    }

    unsigned interface = interfaceAddress(it->second);
    if (!interface) {
        logging::error("Error in BldApp::handleConnect");
        logging::error("Failed to lookup multicast interface %s",it->second.c_str());
        _error("connect", msg, std::string("Failed to lookup multicast interface"));
        return;
    }

    m_det->nodeId = m_drp->nodeId();
    m_det->connect(msg, std::to_string(getId()));

    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void BldApp::handleDisconnect(const json& msg)
{
    // Carry out the queued Unconfigure, if there was one
    if (m_unconfigure) {
        _unconfigure();
    }

    _disconnect();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void BldApp::handlePhase1(const json& msg)
{
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in BldDetectorApp", key.c_str());

    Xtc& xtc = m_det->transitionXtc();
    xtc = {{TypeId::Parent, 0}, {m_det->nodeId}};
    auto bufEnd = m_det->trXtcBufEnd();

    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});

    if (key == "configure") {
        if (m_unconfigure) {
            _unconfigure();
        }

        std::string errorMsg = m_drp->configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::configure";
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp->runInfoSupport(xtc, bufEnd, m_det->namesLookup());
        m_drp->chunkInfoSupport(xtc, bufEnd, m_det->namesLookup());
    }
    else if (key == "unconfigure") {
        // "Queue" unconfiguration until after phase 2 has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        RunInfo runInfo;
        std::string errorMsg = m_drp->beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp->runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp->endrun(phase1Info);
        if (!errorMsg.empty()) {
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }
    }
    else if (key == "enable") {
        bool chunkRequest;
        ChunkInfo chunkInfo;
        std::string errorMsg = m_drp->enable(phase1Info, chunkRequest, chunkInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        } else if (chunkRequest) {
            logging::debug("handlePhase1 enable found chunkRequest");
            m_drp->chunkInfoData(xtc, bufEnd, m_det->namesLookup(), chunkInfo);
        }
        unsigned error = m_det->enable(xtc, bufEnd, phase1Info);
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::enable()";
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        logging::debug("handlePhase1 enable complete");
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void BldApp::handleReset(const json& msg)
{
    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    _unconfigure();
    _disconnect();
    connectionShutdown();
}

} // namespace Drp


int main(int argc, char* argv[])
{
    Drp::Parameters para;
    std::string kwargs_str;
    int c;
    while((c = getopt(argc, argv, "l:p:o:C:b:d:D:u:P:T::k:M:v")) != EOF) {
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
                para.detType = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'k':
                kwargs_str = kwargs_str.empty()
                           ? optarg
                           : kwargs_str + "," + optarg;
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            case 'v':
                ++para.verbose;
                break;
            default:
                return 1;
        }
    }

    switch (para.verbose) {
      case 0:  logging::init(para.instrument.c_str(), LOG_INFO);   break;
      default: logging::init(para.instrument.c_str(), LOG_DEBUG);  break;
    }
    logging::info("logging configured");
    if (optind < argc)
    {
        logging::error("Unrecognized argument:");
        while (optind < argc)
            logging::error("  %s ", argv[optind++]);
        return 1;
    }
    if (para.instrument.empty()) {
        logging::warning("-P: instrument name is missing");
    }
    // Check required parameters
    if (para.partition == unsigned(-1)) {
        logging::critical("-p: partition is mandatory");
        return 1;
    }
    if (para.device.empty()) {
        logging::critical("-d: device is mandatory");
        return 1;
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        return 1;
    }

    // Only one lane is supported by this DRP
    if (std::bitset<PGP_MAX_LANES>(para.laneMask).count() != 1) {
        logging::critical("-l: lane mask must have only 1 bit set");
        return 1;
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        return 1;
    }
    para.detName = "bld";  //para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));
    get_kwargs(kwargs_str, para.kwargs);
    for (const auto& kwargs : para.kwargs) {
        if (kwargs.first == "forceEnet")      continue;
        if (kwargs.first == "ep_fabric")      continue;
        if (kwargs.first == "ep_domain")      continue;
        if (kwargs.first == "ep_provider")    continue;
        if (kwargs.first == "sim_length")     continue;  // XpmDetector
        if (kwargs.first == "timebase")       continue;  // XpmDetector
        if (kwargs.first == "pebbleBufSize")  continue;  // DrpBase
        if (kwargs.first == "pebbleBufCount") continue;  // DrpBase
        if (kwargs.first == "batching")       continue;  // DrpBase
        if (kwargs.first == "directIO")       continue;  // DrpBase
        if (kwargs.first == "pva_addr")       continue;  // DrpBase
        if (kwargs.first == "interface")      continue;
        if (kwargs.first == "timeout")        continue;
        logging::critical("Unrecognized kwarg '%s=%s'\n",
                          kwargs.first.c_str(), kwargs.second.c_str());
        return 1;
    }

    para.maxTrSize = 256 * 1024;
    try {
        Drp::BldApp app(para);
        app.run();
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
}

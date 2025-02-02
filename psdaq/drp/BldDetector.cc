#define __STDC_FORMAT_MACROS 1

#include "BldDetector.hh"
#include "BldNames.hh"

#include <bitset>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "RunInfoDef.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include <getopt.h>
#include <Python.h>
#include <inttypes.h>
#include <time.h>

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

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

    //
    //  Until a PVA gateway can be started on the electron-side
    //

static unsigned getVarDefSize(XtcData::VarDef& vd, const std::vector<unsigned>& as) {
    unsigned sz = 0;
    for(unsigned i=0; i<vd.NameVec.size(); i++) {
        if (as.size()>i && as[i]>0)
            sz += XtcData::Name::get_element_size(vd.NameVec[i].type())*as[i]; // assumes rank=1
        else
            sz += XtcData::Name::get_element_size(vd.NameVec[i].type()); // assumes rank=0
    }
    return sz;
}

BldPVA::BldPVA(std::string det,
               unsigned    interface) : _alg("raw",1,0,0), _interface(interface)
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
    size_t p3 = det.find('+',p2+1);

    _detName = det.substr(   0,     p1).c_str();
    _detType = det.substr(p1+1,p2-p1-1).c_str();
    _detId   = det.substr(p2+1,p3-p2-1).c_str();
    //    unsigned vsn = strtoul(det.substr(p3+1).c_str(),NULL,16);
    //    _alg     = XtcData::Alg("raw",(vsn>>8)&0xf,(vsn>>4)&0xf,(vsn>>0)&0xf);

    std::string sname(_detId);
    // These are CA,PVA for GMD,XGMD but only CA for PCAV
    _pvaAddr    = std::make_shared<Pds_Epics::PVBase>("ca",(sname+":BLD1_MULT_ADDR").c_str());
    _pvaPort    = std::make_shared<Pds_Epics::PVBase>("ca",(sname+":BLD1_MULT_PORT").c_str());
    // This is PVA
    _pvaPayload = std::make_shared<BldDescriptor>    ((sname+":BLD_PAYLOAD"   ).c_str());

    logging::info("BldPVA::BldPVA looking up multicast parameters for %s/%s from %s",
                  _detName.c_str(), _detType.c_str(), _detId.c_str());
}

BldPVA::~BldPVA()
{
}

bool BldPVA::ready() const
{
#define TrueFalse(v) v->ready()?'T':'F'

    logging::debug("%s  addr %c  port %c  payload %c\n",
                   _detId.c_str(),
                   TrueFalse(_pvaAddr),
                   TrueFalse(_pvaPort),
                   TrueFalse(_pvaPayload));
    return (_pvaAddr   ->ready() &&
            _pvaPort   ->ready() &&
            _pvaPayload->ready());
}

unsigned BldPVA::addr() const
{
    unsigned ip = 0;
    in_addr inp;
    if (inet_aton(_pvaAddr->getScalarAs<std::string>().c_str(), &inp)) {
        ip = ntohl(inp.s_addr);
    }
    return ip;
}

unsigned BldPVA::port() const
{
    return _pvaPort->getScalarAs<unsigned>();
}

XtcData::VarDef BldPVA::varDef(unsigned& sz, std::vector<unsigned>& sizes) const
{
    return _pvaPayload->get(sz,sizes);
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

    unsigned mcaddr = 0;
    unsigned mcport = 10148; // 12148, eventually
    uint64_t tscorr = 0x259e9d80UL << 32;
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
        _alg    = XtcData::Alg("raw", 2, 0, 0);
        _varDef.NameVec = BldNames::EBeamDataV7().NameVec;
    }
    else if (strncmp("pcav",name,4)==0) {
        if (name[4]=='h') {
            mcaddr = 0xefff1801;
        }
        else {
            mcaddr = 0xefff1901;
        }
        _alg    = XtcData::Alg("raw", 2, 0, 0);
        _varDef.NameVec = BldNames::PCav().NameVec;
    }
    else if (strncmp("gasdet",name,6)==0) {
        mcaddr = 0xefff1802;
        _alg    = XtcData::Alg("raw", 1, 0, 0);
        _varDef.NameVec = BldNames::GasDet().NameVec;
    }
    else if (strncmp("gmd",name,3)==0) {
        mcaddr = 0xefff1902;
        _alg    = XtcData::Alg("raw", 2, 1, 0);
        _varDef.NameVec = BldNames::GmdV1().NameVec;
    }
    else if (strcmp("xgmd",name)==0) {
        mcaddr = 0xefff1903;
        _alg    = XtcData::Alg("raw", 2, 1, 0);
        _varDef.NameVec = BldNames::GmdV1().NameVec;
    }
    else if ((mcaddr = BldNames::BeamMonitorV1::mcaddr(name))) {
        _detType = _detId = std::string("bmmon");
        _alg    = XtcData::Alg("raw", 1, 0, 0);
        _varDef.NameVec = BldNames::BeamMonitorV1().NameVec;
        _arraySizes = BldNames::BeamMonitorV1().arraySizes();
    }
    else {
        throw std::string("BLD name ")+name+" not recognized";
    }
    unsigned payloadSize = getVarDefSize(_varDef,_arraySizes);
    _handler = std::make_shared<Bld>(mcaddr, mcport, interface,
                                     Bld::DgramTimestampPos, Bld::DgramPulseIdPos,
                                     Bld::DgramHeaderSize, payloadSize,
                                     tscorr);
}

  //
  //  LCLS-II Style
  //
BldFactory::BldFactory(const BldPVA& pva) :
    _detName    (pva.detName()),
    _detType    (pva.detType()),
    _detId      (pva.detId  ()),
    _alg        (pva.alg    ())
{
    while(1) {
        if (pva.ready())
            break;
        usleep(10000);
    }

    unsigned mcaddr = pva.addr();
    unsigned mcport = pva.port();

    unsigned payloadSize = 0;
    _varDef = pva.varDef(payloadSize,_arraySizes);

    _handler = std::make_shared<Bld>(mcaddr, mcport, pva.interface(),
                                     Bld::TimestampPos, Bld::PulseIdPos,
                                     Bld::HeaderSize, payloadSize);
}

BldFactory::BldFactory(const BldFactory& o) :
    _detName    (o._detName),
    _detType    (o._detType),
    _detId      (o._detId),
    _alg        (o._alg)
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

XtcData::NameIndex BldFactory::addToXtc  (XtcData::Xtc& xtc,
                                          const void* bufEnd,
                                          const XtcData::NamesId& namesId)
{
    logging::info("addToXtc %s/%s\n",_detName.c_str(),_detType.c_str());

    XtcData::Names& bldNames = *new(xtc, bufEnd) XtcData::Names(bufEnd,
                                                                _detName.c_str(), _alg,
                                                                _detType.c_str(), _detId.c_str(), namesId);

    bldNames.add(xtc, bufEnd, _varDef);
    return XtcData::NameIndex(bldNames);
}

void BldFactory::addEventData(XtcData::Xtc&          xtc,
                              const void*            bufEnd,
                              XtcData::NamesLookup&  namesLookup,
                              XtcData::NamesId&      namesId)
{
    const Bld& bld = handler();
    XtcData::DescribedData desc(xtc, bufEnd, namesLookup, namesId);
    memcpy(desc.data(), bld.payload(), bld.payloadSize());
    desc.set_data_length(bld.payloadSize());
    unsigned shape[] = {0,0,0,0,0};
    for(unsigned i=0; i<_arraySizes.size(); i++) {
        if (_arraySizes[i]) {
            shape[0] = _arraySizes[i];
            desc.set_array_shape(i,shape);
        }
    }
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

XtcData::VarDef BldDescriptor::get(unsigned& payloadSize, std::vector<unsigned>& sizes)
{
    payloadSize = 0;
    XtcData::VarDef vd;
    const pvd::StructureConstPtr& s = _strct->getStructure();
    if (!s) {
        logging::error("BLD with no payload.  Is FieldMask empty?");
        throw std::string("BLD with no payload.  Is FieldMask empty?");
    }

    const pvd::Structure* structure = static_cast<const pvd::Structure*>(s->getFields()[0].get());

    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    logging::debug("BldDescriptor::get found %u/%u fields", names.size(), fields.size());

    vd.NameVec.push_back(XtcData::Name("severity",XtcData::Name::UINT64));
    payloadSize += 8;

    for (unsigned i=0; i<fields.size(); i++) {
        switch (fields[i]->getType()) {
            case pvd::scalar: {
                const pvd::Scalar* scalar = static_cast<const pvd::Scalar*>(fields[i].get());
                XtcData::Name::DataType type = xtype[scalar->getScalarType()];
                vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type));
                payloadSize += XtcData::Name::get_element_size(type);
                sizes.push_back(0);
                break;
            }

            case pvd::scalarArray: {
                const pvd::ScalarArray* array = static_cast<const pvd::ScalarArray*>(fields[i].get());
                if (array->getArraySizeType()!=pvd::Array::fixed) {
                    throw std::string("PV array type is not Fixed");
                }
                XtcData::Name::DataType type = xtype[array->getElementType()];
                vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type, 1));
                size_t n = array->getMaximumCapacity();
                sizes.push_back(n);
                payloadSize += n*XtcData::Name::get_element_size(type);
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
    m_timestampPos(timestampPos),
    m_pulseIdPos(pulseIdPos),
    m_headerSize(headerSize),
    m_payloadSize(payloadSize),
    m_bufferSize(0),
    m_position(0),
    m_buffer(Bld::MTU),
    m_payload(m_buffer.data()),
    m_timestampCorr(timestampCorr),
    m_pulseId(0),
    m_pulseIdJump(0)
{
    logging::info("Bld listening for %x.%d with payload size %u", mcaddr, port, payloadSize);

    // If mcaddr is non-zero, perform the socket setup.
    if (mcaddr != 0) {
        m_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (m_sockfd < 0)
            HANDLE_ERR("Open socket");

        { 
            unsigned skbSize = 0x1000000;
            if (setsockopt(m_sockfd, SOL_SOCKET, SO_RCVBUF, &skbSize, sizeof(skbSize)) == -1)
                HANDLE_ERR("set so_rcvbuf");
        }

        struct sockaddr_in saddr;
        saddr.sin_family = AF_INET;
        saddr.sin_addr.s_addr = htonl(mcaddr);
        saddr.sin_port = htons(port);
        memset(saddr.sin_zero, 0, sizeof(saddr.sin_zero));
        if (bind(m_sockfd, (struct sockaddr*)&saddr, sizeof(saddr)) < 0)
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
    else {
        // When mcaddr is 0, skip socket setup.
        m_sockfd = -1;
        logging::info("Bld: mcaddr is 0, skipping socket setup.");
    }
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
    // Only close the socket if it was created.
    if (m_sockfd >= 0)
        close(m_sockfd);
}

/*
memory layout for bld packet
header:
uint64_t pulseId
uint64_t timeStamp
uint32_t id
uint64_t severity
uint8_t  payload[]

following events []
uint32_t pulseIdOffset
uint64_t severity
uint8_t  payload[]

*/

//  Read ahead and clear events older than ts (approximate)
// MONA: k-micro might not need this
void     Bld::clear(uint64_t ts)
{
    timespec tts;
    clock_gettime(CLOCK_REALTIME,&tts);
    logging::debug("Bld::clear [%u.%09d]  ts %016llx", tts.tv_sec, tts.tv_nsec, ts);

    uint64_t timestamp(0L);
    uint64_t pulseId  (0L);
    while(1) {
        // get new multicast if buffer is empty
        if ((m_position + m_payloadSize + 4) > m_bufferSize) {
            timestamp = 0;
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

        logging::debug("Bld::clear drop ts %016llx", timestamp);

        unsigned jump = pulseId - m_pulseId;
        m_pulseId = pulseId;
        if (jump != m_pulseIdJump) {
            m_pulseIdJump = jump;
            logging::debug("BLD pulseId jump %u [%u]",jump,pulseId);
        }
    }

    logging::debug("Bld::clear leaving ts %016llx", timestamp);
}

//  Advance to the next event
uint64_t Bld::next()
{
    // MONA: reimplement below to match with k-micro data reading
    // One idea is to have another thread 
    uint64_t timestamp(0L);
    uint64_t pulseId  (0L);

    timespec ts;
    clock_gettime(CLOCK_REALTIME,&ts);

    // get new multicast if buffer is empty
    if ((m_position + m_payloadSize + 4) > m_bufferSize) {
        ssize_t bytes = recv(m_sockfd, m_buffer.data(), Bld::MTU, MSG_DONTWAIT);
        if (bytes <= 0) {
            logging::debug("Bld::next [%u.%09d] no data", ts.tv_sec, ts.tv_nsec);
            return timestamp; // Check only for EWOULDBLOCK and EAGAIN?
        }

        // To do: Handle partial reads?
        m_bufferSize = bytes;
        timestamp    = headerTimestamp();
        pulseId      = headerPulseId  ();
        m_payload    = &m_buffer[m_headerSize];
        m_position   = m_headerSize + m_payloadSize;
        logging::debug("Bld::next [%u.%09d]  ts %016llx  diff %d", ts.tv_sec, ts.tv_nsec, timestamp, ts.tv_sec - (timestamp>>32) - POSIX_TIME_AT_EPICS_EPOCH);
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

    logging::debug("Bld::next timestamp %016llx  pulseId %016llx", timestamp, pulseId);

    return timestamp;
}


// Constructor Implementation
BldDetector::BldDetector(Parameters& para, DrpBase& drp)
    : XpmDetector(&para, &drp.pool) {}

// Override event method (currently empty)
void BldDetector::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) {}


Pgp::Pgp(Parameters& para, DrpBase& drp, Detector* det, bool usePulseId) :
    PgpReader(para, drp.pool, MAX_RET_CNT_C, 32),
    m_para(para), m_drp(drp), m_det(det),
    m_config(0), m_terminate(false), m_running(false),
    m_available(0), m_current(0), m_nDmaRet(0), m_usePulseId(usePulseId)
{
    m_nodeId = det->nodeId;
    if (drp.pool.setMaskBytes(para.laneMask, 0)) {
        logging::error("Failed to allocate lane/vc");
    }
}

Pds::EbDgram* Pgp::_handle(uint32_t& evtIndex)
{
    const Pds::TimingHeader* timingHeader = handle(m_det, m_current);
    if (!timingHeader)  return nullptr;

    uint32_t pgpIndex = timingHeader->evtCounter & (m_pool.nDmaBuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[pgpIndex];

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    evtIndex = event->pebbleIndex;
    XtcData::Src src = m_det->nodeId;
    Pds::EbDgram* dgram = new(m_pool.pebble[evtIndex]) Pds::EbDgram(*timingHeader, src, m_para.rogMask);

    // Collect indices of DMA buffers that can be recycled and reset event
    freeDma(event);

    return dgram;
}

//
static const unsigned TMO_MS = 20;

//  Look at the next timing header (or wait 20ms for the next one)
const Pds::TimingHeader* Pgp::next()
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        auto start = Pds::fast_monotonic_clock::now();
        while (true) {
            m_available = read();
            m_nDmaRet = m_available;
            if (m_available > 0)  break;

            // Time out batches for the TEB
            m_drp.tebContributor().timeout();

            // wait for a total of 10 ms otherwise timeout
            auto now = Pds::fast_monotonic_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > TMO_MS) {
                if (m_running)  logging::debug("pgp timeout");
                return nullptr;
            }
        }
    }

    logging::debug("Pgp::next returning %016llx",
                   m_det->getTimingHeader(dmaIndex[m_current])->time.value());
    return m_det->getTimingHeader(dmaIndex[m_current]);
}

void Pgp::shutdown()
{
    m_terminate.store(true, std::memory_order_release);
    m_det->namesLookup().clear();   // erase all elements
}

void Pgp::worker(std::shared_ptr<Pds::MetricExporter> exporter)
{
    // setup monitoring
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"detseg", std::to_string(m_para.detSegment)},
                                              {"alias", m_para.alias}};
    uint64_t nevents = 0L;
    exporter->add("drp_event_rate", labels, Pds::MetricType::Rate,
                  [&](){return nevents;});
    uint64_t nmissed = 0L;
    exporter->add("bld_miss_count", labels, Pds::MetricType::Counter,
                  [&](){return nmissed;});

    exporter->add("drp_num_dma_ret", labels, Pds::MetricType::Gauge,
                  [&](){return m_nDmaRet;});
    exporter->add("drp_pgp_byte_rate", labels, Pds::MetricType::Rate,
                  [&](){return dmaBytes();});
    exporter->add("drp_dma_size", labels, Pds::MetricType::Gauge,
                  [&](){return dmaSize();});
    exporter->add("drp_th_latency", labels, Pds::MetricType::Gauge,
                  [&](){return latency();});
    exporter->add("drp_num_dma_errors", labels, Pds::MetricType::Gauge,
                  [&](){return nDmaErrors();});
    exporter->add("drp_num_no_common_rog", labels, Pds::MetricType::Gauge,
                  [&](){return nNoComRoG();});
    exporter->add("drp_num_missing_rogs", labels, Pds::MetricType::Gauge,
                  [&](){return nMissingRoGs();});
    exporter->add("drp_num_th_error", labels, Pds::MetricType::Gauge,
                  [&](){return nTmgHdrError();});
    exporter->add("drp_num_pgp_jump", labels, Pds::MetricType::Gauge,
                  [&](){return nPgpJumps();});
    exporter->add("drp_num_no_tr_dgram", labels, Pds::MetricType::Gauge,
                  [&](){return nNoTrDgrams();});

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

    // Initialize the BLD values array. We call it "bldValue" since it may represent
    // either a timestamp or a pulse ID depending on m_usePulseId.
    uint64_t bldValue[m_config.size()];
    memset(bldValue, 0, sizeof(bldValue));
    uint64_t nextId = -1UL;
    for (unsigned i = 0; i < m_config.size(); i++) {
        bldValue[i] = m_config[i]->handler().next();
        if (bldValue[i] < nextId)
            nextId = bldValue[i];
        if (!m_usePulseId)
            logging::info("BldApp::worker Initial timestamp[%d] 0x%" PRIx64, i, bldValue[i]);
        else
            logging::info("BldApp::worker Initial pulseId[%d] 0x%" PRIx64, i, bldValue[i]);
    }

    bool lMissing = false;
    XtcData::NamesLookup& namesLookup = m_det->namesLookup();

    m_tmoState = TmoState::None;
    m_tInitial = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC);

    m_terminate.store(false, std::memory_order_release);

    const Pds::TimingHeader* timingHeader = nullptr;
    uint32_t index = 0;
    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint64_t tts;
        if (!m_usePulseId) {
            // Calculate realtime timeout (50 ms) using timestamp (nanosecond) scale.
            const unsigned TMO_NS = 50000000; // 50,000,000 ns = 50 ms
            timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            // We assume that the timing header timestamp is represented as a 64-bit value
            // where the upper 32 bits are (ts.tv_sec - POSIX_TIME_AT_EPICS_EPOCH) and the lower 32 bits
            // are the nanoseconds.
            if (ts.tv_nsec < TMO_NS) {
                tts = ((ts.tv_sec - POSIX_TIME_AT_EPICS_EPOCH - 1ULL) << 32) |
                      (1000000000ULL + ts.tv_nsec - TMO_NS);
            } else {
                tts = ((ts.tv_sec - POSIX_TIME_AT_EPICS_EPOCH) << 32) |
                      (ts.tv_nsec - TMO_NS);
            }
            logging::debug("tmo time (timestamp mode) %016llx", tts);
        } else {
            // Calculate realtime timeout (50 ms) using pulse ID.
            // Given a pulse rate of 1,000,000 pulses/s, 50 ms corresponds to 50,000 pulses.
            const unsigned TMO_PULSES = 50000; // 50 ms worth of pulses
            timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            // Convert the current time to a pulse id value.
            // We use the same reference epoch as before (POSIX_TIME_AT_EPICS_EPOCH)
            // and assume 1,000,000 pulses per second.
            uint64_t currentPulseId = (ts.tv_sec - POSIX_TIME_AT_EPICS_EPOCH) * 1000000ULL +
                                      (ts.tv_nsec / 1000ULL);
            // The timeout threshold is then the current pulse id minus the number of pulses in 50 ms.
            tts = currentPulseId - TMO_PULSES;
            logging::debug("tmo time (pulse id mode) threshold %016llx", tts);
        }

        //  get oldest timing header
        if (!timingHeader)
            timingHeader = next();

        // For matching, extract the reference value from the timing header.
        if (timingHeader) {
            if (!m_usePulseId) {
                // Use timestamp matching.
                uint64_t ttv = timingHeader->time.value();
                // For each BLD, if its current value is less than the timing header's timestamp,
                // clear and get a new value.
                for (unsigned i = 0; i < m_config.size(); i++) {
                    uint64_t tto = ttv; // We use the timing header's timestamp
                    if (bldValue[i] < tto) {
                        m_config[i]->handler().clear(tto);
                        uint64_t newVal = m_config[i]->handler().next();
                        logging::debug("Bld[%u] replacing timestamp %016llx with %016llx", i, bldValue[i], newVal);
                        bldValue[i] = newVal;
                    }
                    if (bldValue[i] < nextId)
                        nextId = bldValue[i];
                }
            }
            else {
                // Use pulse ID matching.
                uint64_t timingPulseId = timingHeader->pulseId();
                for (unsigned i = 0; i < m_config.size(); i++) {
                    if (bldValue[i] < timingPulseId) {
                        m_config[i]->handler().clear(timingPulseId);
                        uint64_t newVal = m_config[i]->handler().next();
                        logging::debug("Bld[%u] replacing pulseId %016llx with %016llx", i, bldValue[i], newVal);
                        bldValue[i] = newVal;
                    }
                    if (bldValue[i] < nextId)
                        nextId = bldValue[i];
                }
            }
        }

        // Logging for debugging.
        logging::debug("Pgp next common value: 0x%016llx", nextId);

        if (timingHeader) {
            if (timingHeader->service()!=XtcData::TransitionId::L1Accept) {     //  Handle immediately
                Pds::EbDgram* dgram = _handle(index);
                if (!dgram) {
                    m_current++;
                    timingHeader = nullptr;
                    continue;
                }

                // Find the transition dgram in the pool and initialize its header
                Pds::EbDgram* trDgram = m_drp.pool.transitionDgrams[index];
                const void*   bufEnd  = (char*)trDgram + m_para.maxTrSize;
                if (!trDgram)  continue; // Can happen during shutdown
                memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                // copy the temporary xtc created on phase 1 of the transition
                // into the real location
                XtcData::Xtc& trXtc = m_det->transitionXtc();
                trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                switch (dgram->service()) {
                case XtcData::TransitionId::Configure: {
                    logging::info("BLD configure");

                    // Revisit: This is intended to be done by BldDetector::configure()
                    for(unsigned i=0; i<m_config.size(); i++) {
                        XtcData::NamesId namesId(m_nodeId, BldNamesIndex + i);
                        namesLookup[namesId] = m_config[i]->addToXtc(trDgram->xtc, bufEnd, namesId);
                    }
                    break;
                }
                default: {      // Handle other transitions
                    break;
                }
                }

                m_current++;
                timingHeader = nullptr;
                _sendToTeb(*dgram, index);
                nevents++;
            }
            else if ((!m_usePulseId && timingHeader->time.value() < tts)
                     || (m_usePulseId && timingHeader->pulseId() < tts)) {
                // In the "old" case, use the following block:
                Pds::EbDgram* dgram = _handle(index);
                if (!dgram) {
                    m_current++;
                    timingHeader = nullptr;
                    continue;
                }
                const void* bufEnd = (char*)dgram + m_drp.pool.pebble.bufferSize();
                bool lMissed = false;
                for (unsigned i = 0; i < m_config.size(); i++) {
                    if (!m_usePulseId) {
                        // Timestamp matching.
                        if (bldValue[i] == timingHeader->time.value()) {
                            XtcData::NamesId namesId(m_nodeId, BldNamesIndex + i);
                            m_config[i]->addEventData(dgram->xtc, bufEnd, namesLookup, namesId);
                        }
                        else {
                            lMissed = true;
                            if (!lMissing)
                                logging::debug("Missed bld[%u]: pgp %016llx  bld %016llx", i, nextId, bldValue[i]);
                        }
                    }
                    else {
                        // Pulse ID matching.
                        if (bldValue[i] == dgram->pulseId()) {
                            XtcData::NamesId namesId(m_nodeId, BldNamesIndex + i);
                            m_config[i]->addEventData(dgram->xtc, bufEnd, namesLookup, namesId);
                        }
                        else {
                            lMissed = true;
                            if (!lMissing)
                                logging::debug("Missed bld[%u]: pgp %016llx  bld pulseId %016llx", i, nextId, dgram->pulseId());
                        }
                    }
                }
                if (lMissed) {
                    lMissing = true;
                    dgram->xtc.damage.increase(XtcData::Damage::DroppedContribution);
                    nmissed++;
                }
                else {
                    if (lMissing)
                        logging::debug("Found bld: %016lx  %014lx",nextId, dgram->pulseId());
                    lMissing = false;
                }

                m_current++;
                timingHeader = nullptr;
                _sendToTeb(*dgram, index);
                nevents++;
            }
        }
        else {   // No timing header waiting
        }
    }

    // Flush the DMA buffers
    flush();

    logging::info("Worker thread finished");
}

void Pgp::_sendToTeb(Pds::EbDgram& dgram, uint32_t index)
{
    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = (dgram.service() == XtcData::TransitionId::L1Accept)
                         ? m_drp.pool.pebble.bufferSize()
                         : m_para.maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", XtcData::TransitionId::name(dgram.service()), size, maxSize);
        throw "Dgram overflowed buffer";
    }

    auto l3InpBuf = m_drp.tebContributor().fetch(index);
    Pds::EbDgram* l3InpDg = new(l3InpBuf) Pds::EbDgram(dgram);
    if (l3InpDg->isEvent()) {
        auto triggerPrimitive = m_drp.triggerPrimitive();
        if (triggerPrimitive) { // else this DRP doesn't provide input
            const void* bufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
            triggerPrimitive->event(m_drp.pool, index, dgram.xtc, l3InpDg->xtc, bufEnd); // Produce
        }
    }
    m_drp.tebContributor().process(l3InpDg);
}

BldApp::BldApp(Parameters& para, DrpBase& drp, std::unique_ptr<Detector> detector) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp        (drp),
    m_para       (para),
    m_det        (std::move(detector)),  // Injected detector
    m_unconfigure(false)
{
    if (!m_det) {
        logging::critical("Error! Detector object is null.");
        throw std::runtime_error("Could not create Detector object");
    }

    logging::info("Ready for transitions");
}

BldApp::~BldApp()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));
    
    // Py_Finalize() should not be included here. Python should only be 
    // finalized when absolutely necessary 
}

void BldApp::_disconnect()
{
    m_drp.disconnect();
    m_det->shutdown();
}

void BldApp::_unconfigure()
{
    m_drp.pool.shutdown();  // Release Tr buffer pool
    m_drp.unconfigure();    // TebContributor must be shut down before the worker
    if (m_pgp) {
        if (m_exporter)  m_exporter.reset();
        m_pgp->shutdown();
        if (m_workerThread.joinable()) {
            m_workerThread.join();
        }
        m_pgp.reset();
    }
    m_unconfigure = false;
}

json BldApp::connectionInfo(const nlohmann::json& msg)
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo(msg);
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo(ip);
    body["connect_info"].update(bufInfo);
    return body;
}

void BldApp::connectionShutdown()
{
    if (m_det) {
        m_det->connectionShutdown();
    }
    m_drp.shutdown();
    if (m_exporter) {
        m_exporter.reset();
    }
}

void BldApp::_error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void BldApp::handleConnect(const nlohmann::json& msg)
{
    std::string errorMsg = m_drp.connect(msg, getId());
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

    m_det->nodeId = m_drp.nodeId();
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

    XtcData::Xtc& xtc = m_det->transitionXtc();
    xtc = {{XtcData::TypeId::Parent, 0}, {m_det->nodeId}};
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

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_pgp = std::make_unique<Pgp>(m_para, m_drp, m_det.get());

        // Provide EbReceiver with the Detector interface so that additional
        // data blocks can be formatted into the XTC, e.g. trigger information
        m_drp.ebReceiver().configure(m_det.get(), m_pgp.get());

        m_exporter = std::make_shared<Pds::MetricExporter>();
        if (m_drp.exposer()) {
            m_drp.exposer()->RegisterCollectable(m_exporter);
        }

        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::configure";
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_workerThread = std::thread{&Pgp::worker, std::ref(*m_pgp), m_exporter};

        m_drp.runInfoSupport(xtc, bufEnd, m_det->namesLookup());
        m_drp.chunkInfoSupport(xtc, bufEnd, m_det->namesLookup());
    }
    else if (key == "unconfigure") {
        // "Queue" unconfiguration until after phase 2 has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        RunInfo runInfo;
        std::string errorMsg = m_drp.beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        m_drp.runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp.endrun(phase1Info);
        if (!errorMsg.empty()) {
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }
    }
    else if (key == "enable") {
        bool chunkRequest;
        ChunkInfo chunkInfo;
        std::string errorMsg = m_drp.enable(phase1Info, chunkRequest, chunkInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        } else if (chunkRequest) {
            logging::debug("handlePhase1 enable found chunkRequest");
            m_drp.chunkInfoData(xtc, bufEnd, m_det->namesLookup(), chunkInfo);
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

void BldApp::handleReset(const nlohmann::json& msg)
{
    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    _unconfigure();
    _disconnect();
    connectionShutdown();
}

} // namespace Drp


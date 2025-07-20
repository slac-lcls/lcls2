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
#include <sys/prctl.h>
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

    //
    //  Until a PVA gateway can be started on the electron-side
    //

static unsigned getVarDefSize(VarDef& vd, const std::vector<unsigned>& as) {
    unsigned sz = 0;
    for(unsigned i=0; i<vd.NameVec.size(); i++) {
        if (as.size()>i && as[i]>0)
            sz += Name::get_element_size(vd.NameVec[i].type())*as[i]; // assumes rank=1
        else
            sz += Name::get_element_size(vd.NameVec[i].type()); // assumes rank=0
    }
    return sz;
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
    //    _alg     = Alg("raw",(vsn>>8)&0xf,(vsn>>4)&0xf,(vsn>>0)&0xf);

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

VarDef BldPVA::varDef(unsigned& sz, std::vector<unsigned>& sizes) const
{
    return _pvaPayload->get(sz,sizes);
}

  //
  //  LCLS-I Style
  //
BldFactory::BldFactory(const char* name,
                       Parameters& para) :
  _alg        ("raw", 2, 0, 0)
{
    logging::debug("BldFactory::BldFactory %s", name);
    unsigned interface = interfaceAddress(para.kwargs["interface"]);

    if (strchr(name,':'))
        name = strrchr(name,':')+1;

    _detName = std::string(name);
    _detType = std::string(name);
    _detId   = std::string(name);

    _varLenArr = false; // Only feespec has variable length arrays

    unsigned mcaddr = 0;
    unsigned mcport = 10148; // 12148, eventually
    uint64_t tscorr = 0x259e9d80UL << 32;
    // Special case for kmicro:
    if (strcmp("kmicro", name) == 0) {
        int measurementTimeMs = para.kwargs.find("measurementTimeMs") == para.kwargs.end()
                              ? 1000
                              : std::stoi(para.kwargs["measurementTimeMs"]);
        const std::string& iniFilePath = para.kwargs.find("measurementTimeMs") == para.kwargs.end()
                                       ? "tdc_gpx3.ini"
                                       : para.kwargs["iniFile"];
        size_t queueCapacity = para.kwargs.find("queueCapacity") == para.kwargs.end()
                         ? 65536
                         : std::stoul(para.kwargs["queueCapacity"]);
        logging::info("Using INI file: %s", iniFilePath);
        logging::info("Measurement time: %d", measurementTimeMs);
        logging::info("Queue capacity %zu", queueCapacity);
        _alg    = XtcData::Alg("raw", 1, 0, 0);
        _varDef.NameVec = BldNames::KMicroscopeV1().NameVec;
        _arraySizes = BldNames::KMicroscopeV1().arraySizes();
        unsigned payloadSize = getVarDefSize(_varDef,_arraySizes);
        _handler = std::make_shared<KMicroscopeBld>(measurementTimeMs, iniFilePath, queueCapacity, payloadSize);
        return;
    }

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
        _varDef.NameVec = BldNames::EBeamDataV7().NameVec;
    }
    else if (strncmp("pcav",name,4)==0) {
        if (name[4]=='h') {
            mcaddr = 0xefff1801;
        }
        else {
            mcaddr = 0xefff1901;
        }
        _alg    = Alg("raw", 2, 0, 0);
        _varDef.NameVec = BldNames::PCav().NameVec;
    }
    else if (strncmp("gasdet",name,6)==0) {
        mcaddr = 0xefff1802;
        _alg    = Alg("raw", 1, 0, 0);
        _varDef.NameVec = BldNames::GasDet().NameVec;
    }
    else if (strncmp("gmd",name,3)==0) {
        mcaddr = 0xefff1902;
        _alg    = Alg("raw", 2, 1, 0);
        _varDef.NameVec = BldNames::GmdV1().NameVec;
    }
    else if (strcmp("xgmd",name)==0) {
        mcaddr = 0xefff1903;
        _alg    = Alg("raw", 2, 1, 0);
        _varDef.NameVec = BldNames::GmdV1().NameVec;
    }
    else if ((mcaddr = BldNames::BeamMonitorV1::mcaddr(name))) {
        _detType = _detId = std::string("bmmon");
        _alg    = Alg("raw", 1, 0, 0);
        _varDef.NameVec = BldNames::BeamMonitorV1().NameVec;
        _arraySizes = BldNames::BeamMonitorV1().arraySizes();
    }
    else if (strcmp("feespec",name) == 0) {
        mcaddr = 0xefff182e;
        _alg   = XtcData::Alg("raw", 1, 0, 0);
        _varDef.NameVec = BldNames::SpectrometerDataV1().NameVec;
        _arraySizeMap = BldNames::SpectrometerDataV1().arraySizeMap();
        _entryByteSizes = BldNames::SpectrometerDataV1().byteSizes();
        _varLenArr = true; // Feespec arrays are variable length
    }
    else if ((mcaddr = BldNames::UsdUsbDataV1::mcaddr(name))) {
        _detType = _detId = std::string("usdusb");
        _alg   = XtcData::Alg("raw", 1, 0, 0);
        _varDef.NameVec = BldNames::UsdUsbDataV1().NameVec;
        _arraySizes = BldNames::UsdUsbDataV1().arraySizes();
    }
    else {
        throw std::string("BLD name ")+name+" not recognized";
    }
    unsigned payloadSize = _varLenArr ? 0 : getVarDefSize(_varDef,_arraySizes);
    _handler = std::make_shared<Bld>(mcaddr, mcport, interface,
                                     Bld::DgramTimestampPos, Bld::DgramPulseIdPos,
                                     Bld::DgramHeaderSize, payloadSize,
                                     tscorr, _varLenArr, _entryByteSizes, _arraySizeMap);
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

BldBase& BldFactory::handler()
{
    return *_handler;
}

NameIndex BldFactory::addToXtc  (Xtc& xtc,
                                 const void* bufEnd,
                                 const NamesId& namesId)
{
    logging::info("addToXtc %s/%s\n",_detName.c_str(),_detType.c_str());

    Names& bldNames = *new(xtc, bufEnd) Names(bufEnd,
                                              _detName.c_str(), _alg,
                                              _detType.c_str(), _detId.c_str(), namesId);

    bldNames.add(xtc, bufEnd, _varDef);
    return NameIndex(bldNames);
}

void BldFactory::addEventData(Xtc&          xtc,
                              const void*   bufEnd,
                              NamesLookup&  namesLookup,
                              NamesId&      namesId,
                              Parameters& para)
{
    const BldBase& bld = handler();
    DescribedData desc(xtc, bufEnd, namesLookup, namesId);
    // Assume payloadSize has been updated by the time this is called for varLenArr bld

    // Special case for KMicroscope
    if (strcmp("kmicro", para.detType.c_str()) == 0){
        const Drp::KMicroscopeBld& kmBld = static_cast<const Drp::KMicroscopeBld&>(bld);
        new (desc.data()) Drp::KMicroscopeData(kmBld.getMostRecentEvent());

        desc.set_data_length(sizeof(KMicroscopeData));

        unsigned shape[1] = { KMicroscopeData::MAX_EVENTS }; // 16 elements
        for(unsigned i=0; i<_arraySizes.size(); i++) {
            desc.set_array_shape(i, shape);
        }
    } else {
        memcpy(desc.data(), bld.payload(), bld.payloadSize());
        desc.set_data_length(bld.payloadSize());
        unsigned shape[] = {0,0,0,0,0};
    if (!_varLenArr) {
            for(unsigned i=0; i<_arraySizes.size(); i++) {
                if (_arraySizes[i]) {
                    shape[0] = _arraySizes[i];
                    desc.set_array_shape(i,shape);
            }
        }
    } else {
        for (unsigned i=0; i<_arraySizeMap.size(); i++) {
            if (_arraySizeMap[i] != i) {
                unsigned offset = 0;
                for (unsigned j=0; j < _arraySizeMap[i]; j++) {
                    // This should work if all var lens are at the end
                    // Then shouldn't need to account for variable length part in
                    // offset calculation
                    offset += _entryByteSizes[j];
                }
                // Assume entries that dictate array sizes are uint32_t? Or need generic handling?
                shape[0] = *reinterpret_cast<uint32_t*>(&bld.payload()[offset]);
                desc.set_array_shape(i,shape);
            }
            }
        }
    }
}

void BldFactory::configBld()
{
    _handler->initDevice();
}


BldDescriptor::~BldDescriptor()
{
    logging::debug("~BldDescriptor");
}

VarDef BldDescriptor::get(unsigned& payloadSize, std::vector<unsigned>& sizes)
{
    payloadSize = 0;
    VarDef vd;
    const pvd::StructureConstPtr& s = _strct->getStructure();
    if (!s) {
        logging::error("BLD with no payload.  Is FieldMask empty?");
        throw std::string("BLD with no payload.  Is FieldMask empty?");
    }

    const pvd::Structure* structure = static_cast<const pvd::Structure*>(s->getFields()[0].get());

    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    logging::debug("BldDescriptor::get found %u/%u fields", names.size(), fields.size());

    vd.NameVec.push_back(Name("severity",Name::UINT64));
    payloadSize += 8;

    for (unsigned i=0; i<fields.size(); i++) {
        switch (fields[i]->getType()) {
            case pvd::scalar: {
                const pvd::Scalar* scalar = static_cast<const pvd::Scalar*>(fields[i].get());
                Name::DataType type = xtype[scalar->getScalarType()];
                vd.NameVec.push_back(Name(names[i].c_str(), type));
                payloadSize += Name::get_element_size(type);
                sizes.push_back(0);
                break;
            }

            case pvd::scalarArray: {
                const pvd::ScalarArray* array = static_cast<const pvd::ScalarArray*>(fields[i].get());
                if (array->getArraySizeType()!=pvd::Array::fixed) {
                    throw std::string("PV array type is not Fixed");
                }
                Name::DataType type = xtype[array->getElementType()];
                vd.NameVec.push_back(Name(names[i].c_str(), type, 1));
                size_t n = array->getMaximumCapacity();
                sizes.push_back(n);
                payloadSize += n*Name::get_element_size(type);
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

BldBase::BldBase(unsigned mcaddr,
         unsigned port,
         unsigned interface,
         unsigned timestampPos,
         unsigned pulseIdPos,
         unsigned headerSize,
         unsigned payloadSize,
         uint64_t timestampCorr,
         bool     varLenArr,
         std::vector<unsigned> entryByteSizes,
         std::map<unsigned,unsigned> arraySizeMap) :
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
    m_pulseIdJump(0), m_varLenArr(varLenArr),
    m_entryByteSizes(entryByteSizes), m_arraySizeMap(arraySizeMap)
{
    if (m_varLenArr) {
        logging::info("Bld listening for %x.%d with payload size TBD", mcaddr, port);
    } else {
        logging::info("Bld listening for %x.%d with payload size %u",  mcaddr,  port,  payloadSize);
    }

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

BldBase::BldBase(const BldBase& o) :
    m_timestampPos(o.m_timestampPos),
    m_pulseIdPos  (o.m_pulseIdPos),
    m_headerSize  (o.m_headerSize),
    m_payloadSize (o.m_payloadSize),
    m_sockfd      (o.m_sockfd)
{
    logging::error("Bld copy ctor called");
}

BldBase::~BldBase()
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

/**
 * Update m_payloadSize for variable length data arriving from some bld.
 * Inspects the payload based on byte size array and a map for determining
 * what values in the payload determine array lengths elsewhere.
 * Currently assumes values in payload that are also used for array lengths
 * elsewhere in the payload are uint32_t.
 */
void Bld::_calcVarPayloadSize()
{
    if (!m_varLenArr)
        return; // payloadSize is fixed and was passed in at construction, so return

    unsigned offset {0};
    unsigned payloadSize {0};

    for (unsigned i = 0; i < m_arraySizeMap.size(); i++) {
        if (m_arraySizeMap[i] == i) {
            payloadSize += m_entryByteSizes[i];
        } else {
            for (unsigned j=0; j < m_arraySizeMap[i]; j++) {
                // This should work if all var lens are at the end
                // Then shouldn't need to account for variable length part in
                // offset calculation
                offset += m_entryByteSizes[j];
            }
            // Assume entries that dictate array sizes are uint32_t? Or need generic handling?
            uint32_t numEntries = *reinterpret_cast<uint32_t*>(&m_payload[offset]);
            payloadSize += m_entryByteSizes[i]*numEntries;
            offset = 0;
        }
    }

    m_payloadSize = payloadSize;
}

//  Read ahead and clear events older than ts (approximate)
// MONA: k-micro might not need this
void Bld::clear(uint64_t ts)
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
            _calcVarPayloadSize();
            m_position   = m_headerSize + m_payloadSize;
        }
        else if (m_position==0) {
            timestamp    = headerTimestamp();
            if (timestamp >= ts)
                break;
            pulseId      = headerPulseId  ();
            m_payload    = &m_buffer[m_headerSize];
            _calcVarPayloadSize();
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
            _calcVarPayloadSize();
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
        _calcVarPayloadSize();
        m_position   = m_headerSize + m_payloadSize;
        logging::debug("Bld::next [%u.%09d]  ts %016llx  diff %d", ts.tv_sec, ts.tv_nsec, timestamp, ts.tv_sec - (timestamp>>32) - POSIX_TIME_AT_EPICS_EPOCH);
    }
    else if (m_position==0) {
        timestamp    = headerTimestamp();
        pulseId      = headerPulseId  ();
        m_payload    = &m_buffer[m_headerSize];
        _calcVarPayloadSize();
        m_position   = m_headerSize + m_payloadSize;
    }
    else {
        uint32_t timestampOffset = *reinterpret_cast<uint32_t*>(m_buffer.data() + m_position)&0xfffff;
        timestamp   = headerTimestamp() + timestampOffset;
        uint32_t pulseIdOffset   = (*reinterpret_cast<uint32_t*>(m_buffer.data() + m_position)>>20)&0xfff;
        pulseId     = headerPulseId  () + pulseIdOffset;
        m_payload   = &m_buffer[m_position + 4];
        _calcVarPayloadSize();
        m_position += 4 + m_payloadSize;
    }

    logging::debug("Bld::next timestamp %016llx  pulseId %016llx", timestamp, pulseId);

    return timestamp;
}

void Bld::initDevice()
{
    // Initialize device after configure is sent to eb
}

KMicroscopeBld::KMicroscopeBld(int measurementTimeMs,
    const std::string& iniFilePath,
    size_t queueCapacity,
    unsigned payloadSize)
    : BldBase(0, 0, 0, 0, 0, payloadSize, 0),  // These values are unused.
    m_callbackHandler(measurementTimeMs, iniFilePath, queueCapacity)
{
}

KMicroscopeBld::~KMicroscopeBld() {
    // Nothing additional to do; m_callbackHandler cleans up automatically.
}

void KMicroscopeBld::clear(uint64_t ts)
{
    // Reset the stored event. (Additional clearing may be added if needed.)
    m_savedEvent = Drp::KMicroscopeData();
}

uint64_t KMicroscopeBld::next() {
    Drp::KMicroscopeData event;

    auto start_time = std::chrono::steady_clock::now();
    const std::chrono::milliseconds timeout(500);  // Set a timeout of 500ms

    while (!m_callbackHandler.popEvent(event)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));

        auto now = std::chrono::steady_clock::now();
        if (now - start_time > timeout) {
            logging::warning("Timeout exceeded in next(), returning dummy event");
            return 0;  // Return a dummy timestamp to prevent blocking
        }
    }

    // Save (replace) the stored event with the new one.
    m_savedEvent = event;

    // Return only the lower 56 bits of the pulseid.
    uint64_t pulseId = event.pulseid & 0x00ffffffffffffff;
    return pulseId;
}

void KMicroscopeBld::initDevice(){
    m_callbackHandler.init();
    // Start the measurement if not started.
    m_callbackHandler.startMeasurement();
    logging::debug("KMicroscopeBld::initDevice - startMeasurement");
}

// Constructor Implementation
BldDetector::BldDetector(Parameters& para, MemPoolCpu& pool)
    : XpmDetector(&para, &pool) { virtChan = 0; }

// Override event method (currently empty)
void BldDetector::event(Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) {}


Pgp::Pgp(Parameters& para, DrpBase& drp, Detector* det) :
    PgpReader(para, drp.pool, MAX_RET_CNT_C, 32),
    m_para(para), m_drp(drp), m_det(det),
    m_config(0), m_terminate(false), m_running(false),
    m_available(0), m_current(0), m_nDmaRet(0)
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

//
static const unsigned TMO_MS = 20;

//  Look at the next timing header (or wait 20ms for the next one)
const TimingHeader* Pgp::next()
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        auto start = fast_monotonic_clock::now();
        while (true) {
            m_available = read();
            m_nDmaRet = m_available;
            if (m_available > 0)  break;

            // Time out batches for the TEB
            m_drp.tebContributor().timeout();

            // wait for a total of 10 ms otherwise timeout
            auto now = fast_monotonic_clock::now();
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

int Pgp::_setupMetrics(const std::shared_ptr<MetricExporter> exporter)
{
    // setup monitoring
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"detseg", std::to_string(m_para.detSegment)},
                                              {"alias", m_para.alias}};
    m_nevents = 0L;

    // This block is for debugging no. of matched and missed events
    uint64_t matchedEvents = 0;
    uint64_t missedEvents = 0;
    uint64_t last_nevents = 0;
    uint64_t last_matched = 0;
    uint64_t last_missed = 0;
    constexpr uint64_t printInterval = 100000;  // Print every this many events

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

void Pgp::worker(const std::shared_ptr<MetricExporter> exporter)
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
                                                                m_para));
        }
        else if (pvpos > curr && pvpos < next)
            bldPva.push_back(std::make_shared<BldPVA>(s.substr(curr,next-curr),
                                                      interface));
        else
            m_config.push_back(std::make_shared<BldFactory>(s.substr(curr,next-curr).c_str(),
                                                            m_para));
    }

    for(unsigned i=0; i<bldPva.size(); i++)
        m_config.push_back(std::make_shared<BldFactory>(*bldPva[i].get()));

    // Initialize the BLD values array. We call it "bldValue" since it may represent
    // either a timestamp or a pulse ID depending on usePulseId.
    bool usePulseId = strcmp("kmicro", m_para.detType.c_str()) == 0 ? true : false;
    uint64_t bldValue[m_config.size()];
    memset(bldValue, 0, sizeof(bldValue));
    uint64_t nextId = -1UL;
    //for (unsigned i = 0; i < m_config.size(); i++) {
    //    logging::info("BldApp::worker config: %u detType: %s calling next()", i, m_para.detType.c_str());
    //    bldValue[i] = m_config[i]->handler().next();
    //    if (bldValue[i] < nextId)
    //        nextId = bldValue[i];
    //    if (!usePulseId)
    //        logging::info("BldApp::worker Initial timestamp[%d] 0x%" PRIx64, i, bldValue[i]);
    //    else
    //        logging::info("BldApp::worker Initial pulseId[%d] 0x%" PRIx64, i, bldValue[i]);
    //}

    bool lMissing = false;
    NamesLookup& namesLookup = m_det->namesLookup();

    m_tmoState = TmoState::None;
    m_tInitial = fast_monotonic_clock::now(CLOCK_MONOTONIC);

    m_terminate.store(false, std::memory_order_release);

    const Pds::TimingHeader* timingHeader = nullptr;
    uint32_t index = 0;
    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        //  get oldest timing header
        if (!timingHeader)
            timingHeader = next();

        if (timingHeader) {
            if (timingHeader->service()!=TransitionId::L1Accept) {     //  Handle immediately
                EbDgram* dgram = _handle(index);
                if (!dgram) {
                    m_current++;
                    timingHeader = nullptr;
                    continue;
                }

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

                switch (dgram->service()) {
                case TransitionId::Configure: {
                    logging::info("BLD configure");

                    // Revisit: This is intended to be done by BldDetector::configure()
                    for(unsigned i=0; i<m_config.size(); i++) {
                        NamesId namesId(m_det->nodeId, BldNamesIndex + i);
                        namesLookup[namesId] = m_config[i]->addToXtc(trDgram->xtc, bufEnd, namesId);
                        m_config[i]->configBld();
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
                m_nevents++;
            }
            else {
                // Calculate realtime timeout (50 ms)
                const unsigned TMO_NS = 50000000;
                timespec ts;
                clock_gettime(CLOCK_REALTIME,&ts);
                uint64_t tts = ts.tv_sec - POSIX_TIME_AT_EPICS_EPOCH;
                if (ts.tv_nsec < TMO_NS) {
                    tts = (tts-1)<<32;
                    tts |= 1000000000+ts.tv_nsec-TMO_NS;
                }
                else {
                    tts <<= 32;
                    tts |= ts.tv_nsec-TMO_NS;
                }
                logging::debug("tmo time (timestamp mode) %016llx", tts);
                if (usePulseId) {
                    // Extract the parts from the 64-bit timestamp:
                    //   upper 32 bits: seconds since EPICS epoch
                    //   lower 32 bits: nanoseconds
                    uint32_t sec  = tts >> 32;
                    uint32_t nsec = tts & 0xffffffff;
                    double total_sec = sec + nsec / 1e9;
                    // Use the effective rate of about 928300 pulses per second:
                    tts = static_cast<uint64_t>(total_sec * 928300.0);
                    logging::debug("tmo time (pulse id mode) threshold 0x%" PRIx64, tts);
                }

                //  get oldest BLD and throw away anything older than timingheader or timeout value
                nextId = -1UL;
                if (!usePulseId) {
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
                            //m_config[i]->handler().clear(timingPulseId); MONA No need for clearing?
                            uint64_t newVal = m_config[i]->handler().next();
                            logging::debug("Bld[%u] replacing pulseId %016llx with %016llx", i, bldValue[i], newVal);
                            bldValue[i] = newVal;
                        }
                        if (bldValue[i] < nextId)
                            nextId = bldValue[i];
                    }
                }
                logging::debug("Bld next: 0x%" PRIx64, nextId);

                if ((!usePulseId && timingHeader->time.value() < tts)
                     || (usePulseId && timingHeader->pulseId() < tts)) {
                    // In the "old" case, use the following block:
                    EbDgram* dgram = _handle(index);
                    if (!dgram) {
                        m_current++;
                        timingHeader = nullptr;
                        continue;
                    }
                    const void* bufEnd = (char*)dgram + m_drp.pool.pebble.bufferSize();
                    bool lMissed = false;
                    for (unsigned i = 0; i < m_config.size(); i++) {
                        if (!usePulseId) {
                            // Timestamp matching.
                            if (bldValue[i] == timingHeader->time.value()) {
                                NamesId namesId(m_det->nodeId, BldNamesIndex + i);
                                m_config[i]->addEventData(dgram->xtc, bufEnd, namesLookup, namesId, m_para);
                            }
                            else {
                                lMissed = true;
                                if (!lMissing)
                                    logging::debug("Missed bld[%u]: pgp %016lx  bld %016lx  pid %014lx",
                                                   i, timingHeader->time.value(), bldValue[i], dgram->pulseId());
                            }
                        }
                        else {
                            // Pulse ID matching.
                            if (bldValue[i] == timingHeader->pulseId()) {
                                matchedEvents++;
                                XtcData::NamesId namesId(m_nodeId, BldNamesIndex + i);
                                m_config[i]->addEventData(dgram->xtc, bufEnd, namesLookup, namesId, m_para);
                                logging::debug("Found bld[%u]: pgp %016lx  bld %016lx  pid %014lx",
                                                i, timingHeader->pulseId(), bldValue[i], dgram->pulseId());
                            }
                            else {
                                missedEvents++;
                                lMissed = true;
                                if (!lMissing)
                                    logging::debug("Missed bld[%u]: pgp %016lx  bld %016lx  timestamp %016lx",
                                                   i, timingHeader->pulseId(), bldValue[i], timingHeader->time.value());
                            }
                        }
                    }
                    if (lMissed) {
                        lMissing = true;
                        dgram->xtc.damage.increase(Damage::DroppedContribution);
                        m_nmissed++;
                    }
                    else {
                        if (lMissing)
                            logging::debug("Lost bld: %016lx  %014lx",nextId, dgram->pulseId());
                        lMissing = false;
                    }

                    m_current++;
                    timingHeader = nullptr;
                    _sendToTeb(*dgram, index);
                    m_nevents++;
                    // Print every n events
                    if (nevents % printInterval == 0) {
                        // Deltas
                        uint64_t delta_nevents = nevents - last_nevents;
                        uint64_t delta_matched = matchedEvents - last_matched;
                        uint64_t delta_missed  = missedEvents - last_missed;

                        double delta_percentMatched = (delta_nevents > 0) ? (delta_matched / (double)delta_nevents) * 100.0 : 0.0;
                        double delta_percentMissed  = (delta_nevents > 0) ? (delta_missed  / (double)delta_nevents) * 100.0 : 0.0;

                        logging::info("[Pgp::worker] Interval-> Events: %llu (Total: %llu), Matched: %llu (%.2f%%), Missed: %llu (%.2f%%)",
                                    delta_nevents, nevents, delta_matched, delta_percentMatched, delta_missed, delta_percentMissed);

                        // Update last_* for next interval
                        last_nevents = nevents;
                        last_matched = matchedEvents;
                        last_missed  = missedEvents;
                    }

                }  // Done with L1Accept with correct timing
            }  // Done with one event
        }
        else {   // No timing header waiting
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

    // Py_Finalize() should not be included here. Python should only be
    // finalized when absolutely necessary
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

        // Configure the detector first
        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (error) {
            std::string errorMsg = "Phase 1 error in Detector::configure";
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

        // Next, configure the DRP
        std::string errorMsg = m_drp->configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
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
    while((c = getopt(argc, argv, "p:o:C:b:d:D:u:P:T::k:M:v")) != EOF) {
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
        if (kwargs.first == "measurementTimeMs") continue;  // K-microscope
        if (kwargs.first == "iniFile") continue;            // K-microscope
        if (kwargs.first == "queueCapacity") continue;      // K-microscope
        logging::critical("Unrecognized kwarg '%s=%s'\n",
                          kwargs.first.c_str(), kwargs.second.c_str());
        return 1;
    }


    /*
    //  Add pva_addr to the environment
    if (para.kwargs.find("pva_addr")!=para.kwargs.end()) {
        const char* a = para.kwargs["pva_addr"].c_str();
        char* p = getenv("EPICS_PVA_ADDR_LIST");
        char envBuff[256];
        if (p)
            sprintf(envBuff,"EPICS_PVA_ADDR_LIST=%s %s", p, a);
        else
            sprintf(envBuff,"EPICS_PVA_ADDR_LIST=%s", a);
        logging::info("Setting env %s\n", envBuff);
        putenv(envBuff);
    }
    */

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

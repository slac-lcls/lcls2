#include "UdpEncoder.hh"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <getopt.h>
#include <cassert>
#include <bitset>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <map>
#include <algorithm>
#include <limits>
#include <thread>
#include <Python.h>
#include <arpa/inet.h>
#include "DataDriver.h"
#include "RunInfoDef.hh"
#include "xtcdata/xtc/Damage.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

static const XtcData::TimeStamp TimeMax(std::numeric_limits<unsigned>::max(),
                                        std::numeric_limits<unsigned>::max());
static unsigned tsMatchDegree = 0;

// forward declarations
int setrcvbuf(int socketFd, unsigned size);
int createUdpSocket(int port);

class RawDef:public XtcData::VarDef
{
public:
  enum index
    {
      encoderValue,
      frameCount,
      timing,
      scale,
      scaleDenom,
      mode,
      error,
      majorVersion,
      minorVersion,
      microVersion,
      hardwareID
    };

  RawDef()
   {
       NameVec.push_back({"encoderValue", XtcData::Name::UINT32,1});
       // frameCount is common to all channels
       NameVec.push_back({"frameCount", XtcData::Name::UINT16});
       NameVec.push_back({"timing", XtcData::Name::UINT32,1});
       NameVec.push_back({"scale", XtcData::Name::UINT16,1});
       NameVec.push_back({"scaleDenom", XtcData::Name::UINT16,1});
       NameVec.push_back({"mode", XtcData::Name::UINT8,1});
       NameVec.push_back({"error", XtcData::Name::UINT8,1});
       NameVec.push_back({"majorVersion", XtcData::Name::UINT16,1});
       NameVec.push_back({"minorVersion", XtcData::Name::UINT8,1});
       NameVec.push_back({"microVersion", XtcData::Name::UINT8,1});
       NameVec.push_back({"hardwareID", XtcData::Name::CHARSTR,1});
   }
} RawDef;

namespace Drp {

static const XtcData::Name::DataType xtype[] = {
  XtcData::Name::UINT8 , // pvBoolean
  XtcData::Name::INT8  , // pvByte
  XtcData::Name::INT16 , // pvShort
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

bool UdpMonitor::ready(UdpEncoder* udpDetector)
{
    m_udpDetector = udpDetector;

    return true;   // TODO
}

void UdpMonitor::onConnect()
{
    logging::info("%s connected", name().c_str());
}

void UdpMonitor::onDisconnect()
{
    logging::info("%s disconnected", name().c_str());
}

class Pgp
{
public:
    Pgp(const Parameters& para, DrpBase& drp, const bool& running) :
        m_para(para), m_pool(drp.pool), m_tebContributor(drp.tebContributor()), m_running(running),
        m_available(0), m_current(0), m_lastComplete(0), m_latency(0), m_nDmaRet(0)
    {
        m_nodeId = drp.nodeId();
        if (drp.pool.setMaskBytes(para.laneMask, 0)) {
            logging::error("Failed to allocate lane/vc");
        }
    }

    Pds::EbDgram* next(uint32_t& evtIndex, uint64_t& bytes);
    const int64_t latency() { return m_latency; }
    const uint64_t nDmaRet() { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex, uint64_t& bytes);
    const Parameters& m_para;
    MemPool& m_pool;
    Pds::Eb::TebContributor& m_tebContributor;
    static const int MAX_RET_CNT_C = 100;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    const bool& m_running;
    int32_t m_available;
    int32_t m_current;
    uint32_t m_lastComplete;
    XtcData::TransitionId::Value m_lastTid;
    uint32_t m_lastData[6];
    unsigned m_nodeId;
    int64_t m_latency;
    uint64_t m_nDmaRet;
};

Pds::EbDgram* Pgp::_handle(uint32_t& current, uint64_t& bytes)
{
    uint32_t size = dmaRet[m_current];
    uint32_t index = dmaIndex[m_current];
    uint32_t lane = (dest[m_current] >> 8) & 7;
    bytes += size;
    if (size > m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d", size, m_pool.dmaSize());
        throw "DMA overflowed buffer";
    }

    const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
    uint32_t evtCounter = data[5] & 0xffffff;
    const unsigned bufferMask = m_pool.nbuffers() - 1;
    current = evtCounter & bufferMask;
    PGPEvent* event = &m_pool.pgpEvents[current];
    // Revisit: Doesn't always work?  assert(event->mask == 0);

    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    event->mask |= (1 << lane);

    logging::debug("PGPReader  lane %u  size %u  hdr %016lx.%016lx.%08x",
                   lane, size,
                   reinterpret_cast<const uint64_t*>(data)[0],
                   reinterpret_cast<const uint64_t*>(data)[1],
                   reinterpret_cast<const uint32_t*>(data)[4]);

    const Pds::TimingHeader* timingHeader = reinterpret_cast<const Pds::TimingHeader*>(data);
    if (timingHeader->error()) {
        logging::error("Timing header error bit is set");
    }
    XtcData::TransitionId::Value transitionId = timingHeader->service();
    if (transitionId != XtcData::TransitionId::L1Accept) {
        if ( transitionId != XtcData::TransitionId::SlowUpdate) {
            logging::info("PGPReader  saw %s @ %u.%09u (%014lx)",
                          XtcData::TransitionId::name(transitionId),
                          timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                          timingHeader->pulseId());
        }
        else {
            logging::debug("PGPReader  saw %s @ %u.%09u (%014lx)",
                           XtcData::TransitionId::name(transitionId),
                           timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                           timingHeader->pulseId());
        }
        if (transitionId == XtcData::TransitionId::BeginRun) {
            m_lastComplete = 0;  // EvtCounter reset
        }
    }
    if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
        logging::critical("%sPGPReader: Jump in complete l1Count %u -> %u | difference %d, tid %s%s",
                          RED_ON, m_lastComplete, evtCounter, evtCounter - m_lastComplete, XtcData::TransitionId::name(transitionId), RED_OFF);
        logging::critical("data: %08x %08x %08x %08x %08x %08x",
                          data[0], data[1], data[2], data[3], data[4], data[5]);

        logging::critical("lastTid %s", XtcData::TransitionId::name(m_lastTid));
        logging::critical("lastData: %08x %08x %08x %08x %08x %08x",
                          m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5]);

        throw "Jump in event counter";

        for (unsigned e=m_lastComplete+1; e<evtCounter; e++) {
            PGPEvent* brokenEvent = &m_pool.pgpEvents[e & bufferMask];
            logging::error("broken event:  %08x", brokenEvent->mask);
            brokenEvent->mask = 0;

        }
    }
    m_lastComplete = evtCounter;
    m_lastTid = transitionId;
    memcpy(m_lastData, data, 24);

    auto now = std::chrono::system_clock::now();
    auto dgt = std::chrono::seconds{timingHeader->time.seconds() + POSIX_TIME_AT_EPICS_EPOCH}
             + std::chrono::nanoseconds{timingHeader->time.nanoseconds()};
    std::chrono::system_clock::time_point tp{std::chrono::duration_cast<std::chrono::system_clock::duration>(dgt)};
    m_latency = std::chrono::duration_cast<ms_t>(now - tp).count();

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    Pds::EbDgram* dgram = new(m_pool.pebble[current]) Pds::EbDgram(*timingHeader, XtcData::Src(m_nodeId), m_para.rogMask);

    return dgram;
}

Pds::EbDgram* Pgp::next(uint32_t& evtIndex, uint64_t& bytes)
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        auto start = std::chrono::steady_clock::now();
        while (true) {
            m_available = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
            m_nDmaRet = m_available;
            if (m_available > 0) {
                m_pool.allocate(m_available);
                break;
            }

            // wait for a total of 10 ms otherwise timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > 10) {
                //if (m_running)  logging::debug("pgp timeout");
                return nullptr;
            }
        }
    }

    Pds::EbDgram* dgram = _handle(evtIndex, bytes);
    m_current++;
    return dgram;
}


UdpEncoder::UdpEncoder(Parameters& para, std::shared_ptr<UdpMonitor>& udpMonitor, DrpBase& drp) :
    XpmDetector     (&para, &drp.pool),
    m_drp           (drp),
    m_udpMonitor    (udpMonitor),
    m_pgpQueue      (drp.pool.nbuffers()),
    m_pvQueue       (8),                  // Revisit size
    m_bufferFreelist(m_pvQueue.size()),
    m_terminate     (false),
    m_running       (false),
    m_resetHwCount  (true),
    m_outOfOrder    (false),
    m_missingData   (false),
    m_notifySocket{&m_context, ZMQ_PUSH}
{
    // ZMQ socket for reporting errors
    m_notifySocket.connect({"tcp://" + m_para->collectionHost + ":" + std::to_string(CollectionApp::zmq_base_port + m_para->partition)});

    // UDP socket for receiving data
    int dataPort = (m_para->loopbackPort) ? m_para->loopbackPort : DefaultDataPort;
    _dataFd = createUdpSocket(dataPort);
    logging::debug("createUdpSocket(%d) returned %d", dataPort, _dataFd);
}

UdpEncoder::~UdpEncoder()
{
    if (_dataFd > 0) {
        close(_dataFd);
    }
}

void UdpEncoder::addNames(unsigned segment, XtcData::Xtc& xtc, const void* bufEnd)
{
    XtcData::Alg encoderRawAlg("raw", UdpEncoder::MajorVersion, UdpEncoder::MinorVersion, UdpEncoder::MicroVersion);
    XtcData::NamesId rawNamesId(nodeId, segment);
    XtcData::Names&  rawNames = *new(xtc, bufEnd) XtcData::Names(bufEnd,
                                                                 m_para->detName.c_str(), encoderRawAlg,
                                                                 m_para->detType.c_str(), m_para->serNo.c_str(), rawNamesId, segment);
    rawNames.add(xtc, bufEnd, RawDef);
    m_namesLookup[rawNamesId] = XtcData::NameIndex(rawNames);
}

  //std::string UdpEncoder::sconfigure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
unsigned UdpEncoder::configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
{
    logging::debug("entered %s", __PRETTY_FUNCTION__);

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    if (m_exporter)  m_exporter.reset();
    m_exporter = std::make_shared<Pds::MetricExporter>();
    if (m_drp.exposer()) {
        m_drp.exposer()->RegisterCollectable(m_exporter);
    }

    if (!m_udpMonitor->ready(this)) {
        std::string error = "Failed to connect with " + m_udpMonitor->name();
        logging::error(error.c_str());
        //return error;
        return 1;
    }

    addNames(0, xtc, bufEnd);

    size_t bufSize = m_pool->pebble.bufferSize();
    m_buffer.resize(m_pvQueue.size() * bufSize);
    for(unsigned i = 0; i < m_pvQueue.size(); ++i) {
        m_bufferFreelist.push(reinterpret_cast<XtcData::Dgram*>(&m_buffer[i * bufSize]));
    }

    m_resetHwCount = true;

    m_terminate.store(false, std::memory_order_release);

    m_workerThread = std::thread{&UdpEncoder::_worker, this};
    m_udpReceiverThread = std::thread{&UdpEncoder::_udpReceiver, this};

    return 0;
}

unsigned UdpEncoder::unconfigure()
{
    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    if (m_udpReceiverThread.joinable()) {
        m_udpReceiverThread.join();
    }
    m_udpMonitor->clear();
    m_namesLookup.clear();   // erase all elements

    return 0;
}

void UdpEncoder::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* pgpEvent)
{
    encoder_frame_t frame;
    bool missing = false;

    // read from the udp socket that triggered select()
    int rv = _readFrame(&frame, missing);

    // if reading frame failed, record damage and return early
    if (rv) {
        dgram.xtc.damage.increase(XtcData::Damage::UserDefined);
        logging::critical("%s: failed to read UDP frame", __PRETTY_FUNCTION__);
        return;
    }

    // if missing data reported, record damage
    if (missing) {
        // record damage
        dgram.xtc.damage.increase(XtcData::Damage::MissingData);
        // report first occurance
        if (!getMissingData()) {
            char errmsg[128];
            snprintf(errmsg, sizeof(errmsg),
                     "Missing data for frame %hu", frame.header.frameCount);
            setMissingData(errmsg);
        }
    }

    logging::debug("%s: frame=%hu  encoderValue=%u  timing=%u  scale=%u  scaleDenom=%u  mode=%u  error=%u  version=%u.%u.%u",
                   __PRETTY_FUNCTION__,
                   frame.header.frameCount,
                   frame.channel[0].encoderValue,
                   frame.channel[0].timing,
                   (unsigned) frame.channel[0].scale,
                   (unsigned) frame.channel[0].scaleDenom,
                   (unsigned) frame.channel[0].mode,
                   (unsigned) frame.channel[0].error,
                   (unsigned) frame.header.majorVersion,
                   (unsigned) frame.header.minorVersion,
                   (unsigned) frame.header.microVersion);

    // reset frame counter
    if (m_resetHwCount) {
        m_count = 0;
        m_countOffset = frame.header.frameCount - 1;
        m_resetHwCount = false;
    }

    // update frame counter
    uint16_t stuck16 = (uint16_t)(m_count + m_countOffset);
    ++m_count;
    uint16_t sum16 = (uint16_t)(m_count + m_countOffset);

    if (!getOutOfOrder()) {
        char errmsg[128];
        // check for out-of-order condition
        if (frame.header.frameCount == stuck16) {
            snprintf(errmsg, sizeof(errmsg),
                     "Out-of-order: frame count %hu repeated in consecutive frames", stuck16);
            setOutOfOrder(errmsg);

        } else if (frame.header.frameCount != sum16) {
            snprintf(errmsg, sizeof(errmsg),
                     "Out-of-order: hw count (%hu) != sw count (%hu) + offset (%u) == (%hu)",
                     frame.header.frameCount, m_count, m_countOffset, sum16);
            setOutOfOrder(errmsg);
        }
    }

    // record damage
    if (m_outOfOrder) {
        dgram.xtc.damage.increase(XtcData::Damage::OutOfOrder);
    }

    // ----- CreateData  ------------------------------------------------------
    unsigned segment = 0;

    XtcData::NamesId namesId1(nodeId, segment);
    XtcData::CreateData raw(dgram.xtc, bufEnd, m_namesLookup, namesId1);
    unsigned shape[XtcData::MaxRank] = {1};

    // ...encoderValue
    XtcData::Array<uint32_t> arrayA = raw.allocate<uint32_t>(RawDef::encoderValue,shape);
    arrayA(0) = frame.channel[0].encoderValue;

    // ...frameCount
    raw.set_value(RawDef::frameCount, frame.header.frameCount);

    // ...timing
    XtcData::Array<uint32_t> arrayB = raw.allocate<uint32_t>(RawDef::timing,shape);
    arrayB(0) = frame.channel[0].timing;

    // ...scale
    XtcData::Array<uint16_t> arrayC = raw.allocate<uint16_t>(RawDef::scale,shape);
    arrayC(0) = frame.channel[0].scale;

    // ...scaleDenom
    XtcData::Array<uint16_t> arrayJ = raw.allocate<uint16_t>(RawDef::scaleDenom,shape);
    arrayJ(0) = frame.channel[0].scaleDenom;

    // ...mode
    XtcData::Array<uint8_t> arrayD = raw.allocate<uint8_t>(RawDef::mode,shape);
    arrayD(0) = frame.channel[0].mode;

    // ...error
    XtcData::Array<uint8_t> arrayE = raw.allocate<uint8_t>(RawDef::error,shape);
    arrayE(0) = frame.channel[0].error;

    // ...majorVersion
    XtcData::Array<uint16_t> arrayF = raw.allocate<uint16_t>(RawDef::majorVersion,shape);
    arrayF(0) = frame.header.majorVersion;

    // ...minorVersion
    XtcData::Array<uint8_t> arrayG = raw.allocate<uint8_t>(RawDef::minorVersion,shape);
    arrayG(0) = frame.header.minorVersion;

    // ...microVersion
    XtcData::Array<uint8_t> arrayH = raw.allocate<uint8_t>(RawDef::microVersion,shape);
    arrayH(0) = frame.header.microVersion;

    // ...hardwareID
    char buf[16];
    snprintf(buf, sizeof(buf), "%s", frame.header.hardwareID);
    raw.set_string(RawDef::hardwareID, buf);
}

void UdpEncoder::_loopbackInit()
{
    logging::debug("%s (port = %d)", __PRETTY_FUNCTION__, m_para->loopbackPort);

    if (m_para->loopbackPort > 0) {
        m_loopbackFd = socket(AF_INET,SOCK_DGRAM, 0);
        if (m_loopbackFd == -1) {
            perror("socket");
            logging::error("failed to create loopback socket");
        }

        bzero(&m_loopbackAddr, sizeof(m_loopbackAddr));
        m_loopbackAddr.sin_family = AF_INET;
        m_loopbackAddr.sin_addr.s_addr=inet_addr("127.0.0.1");
        m_loopbackAddr.sin_port=htons(m_para->loopbackPort);
    }

}

void UdpEncoder::_loopbackFini()
{
    logging::debug("%s", __PRETTY_FUNCTION__);

    if (m_loopbackFd > 0) {
        if (close(m_loopbackFd)) {
            logging::error("failed to close loopback socket");
        }
    }
}

void UdpEncoder::_loopbackSend()
{
    char mybuf[sizeof(encoder_header_t) + sizeof(encoder_channel_t)];
    memset((void *)mybuf, 0, sizeof(mybuf));

    encoder_header_t *pHeader = (encoder_header_t *)mybuf;
    encoder_channel_t *pChannel = (encoder_channel_t *)(pHeader + 1);

    ++ m_loopbackFrameCount;     // advance the simulated frame counter
    pHeader->frameCount = htons(m_loopbackFrameCount);
    pHeader->majorVersion = htons(UdpEncoder::MajorVersion);
    pHeader->minorVersion = UdpEncoder::MinorVersion;
    pHeader->microVersion = UdpEncoder::MicroVersion;
    pHeader->channelMask = 0x01;
    sprintf(pHeader->hardwareID, "%s", "LOOPBACK SIM");

    pChannel->encoderValue = htonl(170000);
    pChannel->timing = htonl(54321);
    pChannel->scale = htons(1);
    pChannel->scaleDenom = htons(150);

    int sent = sendto(m_loopbackFd, (void *)mybuf, sizeof(mybuf), 0,
                  (struct sockaddr *)&m_loopbackAddr, sizeof(m_loopbackAddr));

    if (sent == -1) {
        perror("sendto");
        logging::error("failed to send to loopback socket");
    } else {
        logging::debug("%s: sent = %d", __PRETTY_FUNCTION__, sent);
    }
}

void UdpEncoder::_worker()
{
    logging::info("Worker thread started");

    // setup monitoring
    std::map<std::string, std::string> labels{{"instrument", m_para->instrument},
                                              {"partition", std::to_string(m_para->partition)},
                                              {"detname", m_para->detName},
                                              {"detseg", std::to_string(m_para->detSegment)},
                                              {"alias", m_para->alias}};
    m_nEvents = 0;
    m_exporter->add("drp_event_rate", labels, Pds::MetricType::Rate,
                    [&](){return m_nEvents;});
    uint64_t bytes = 0L;
    m_exporter->add("drp_pgp_byte_rate", labels, Pds::MetricType::Rate,
                    [&](){return bytes;});
    m_nUpdates = 0;
    m_exporter->add("udp_update_rate", labels, Pds::MetricType::Rate,
                    [&](){return m_nUpdates;});
    m_nMatch = 0;
    m_exporter->add("udp_match_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nMatch;});
    m_nEmpty = 0;
    m_exporter->add("udp_empty_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nEmpty;});
    m_nMissed = 0;
    m_exporter->add("udp_miss_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nMissed;});
    m_nTooOld = 0;
    m_exporter->add("udp_tooOld_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nTooOld;});
    m_nTimedOut = 0;
    m_exporter->add("udp_timeout_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nTimedOut;});

    m_exporter->add("drp_worker_input_queue", labels, Pds::MetricType::Gauge,
                    [&](){return m_pgpQueue.guess_size();});
    m_exporter->constant("drp_worker_queue_depth", labels, m_pgpQueue.size());

    // Borrow this for awhile
    m_exporter->add("drp_worker_output_queue", labels, Pds::MetricType::Gauge,
                    [&](){return m_pvQueue.guess_size();});

    Pgp pgp(*m_para, m_drp, m_running);

    m_exporter->add("drp_th_latency", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.latency();});
    m_exporter->add("drp_num_dma_ret", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nDmaRet();});

    const uint64_t msTmo = m_para->kwargs.find("match_tmo_ms") != m_para->kwargs.end()
                         ? std::stoul(m_para->kwargs["match_tmo_ms"])
                         : 100;

    if (m_para->loopbackPort) {
        _loopbackInit();        // LOOPBACK TEST
    }

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        Pds::EbDgram* dgram = pgp.next(index, bytes);
        if (dgram) {
            m_nEvents++;
            logging::debug("Worker thread: m_nEvents = %d", m_nEvents);

            XtcData::TransitionId::Value service = dgram->service();

            if (service == XtcData::TransitionId::L1Accept) {
                if (m_para->loopbackPort) {
                    _loopbackSend();        // LOOPBACK TEST
                }
            }

            // Also queue SlowUpdates to keep things in time order
            if ((service == XtcData::TransitionId::L1Accept) ||
                (service == XtcData::TransitionId::SlowUpdate)) {
                m_pgpQueue.push(index);

                //printf("                         PGP: %u.%09u\n",
                //       dgram->time.seconds(), dgram->time.nanoseconds());

                _matchUp();

                // Prevent PGP events from stacking up by by timing them out.
                // The maximum timeout is < the TEB event build timeout to keep
                // prompt contributions from timing out before latent ones arrive.
                // If the PV is updating, _timeout() never finds anything to do.
                XtcData::TimeStamp timestamp;
                //const uint64_t msTmo = tsMatchDegree==2 ? 100 : 1000; //4400;
                //const uint64_t msTmo = tsMatchDegree!=1 ? 100 : 1141; // 1s + fid. time for fuzzy ts matching
                //const uint64_t ebTmo = 6000; // This overflows PGP (?) buffers: Pds::Eb::EB_TMO_MS/2 - 100;
                //const uint64_t ebTmo = Pds::Eb::EB_TMO_MS/2 - 100; // This overflows PGP (?) buffers
                //const uint64_t msTmo = tsMatchDegree==2 ? 100 : ebTmo;
                const uint64_t nsTmo = msTmo * 1000000;
                _timeout(timestamp.from_ns(dgram->time.to_ns() - nsTmo));
            }
            else {
                // Allocate a transition dgram from the pool and initialize its header
                Pds::EbDgram* trDgram = m_pool->allocateTr();
                const void*   bufEnd  = (char*)trDgram + m_para->maxTrSize;
                *trDgram = *dgram;
                // copy the temporary xtc created on phase 1 of the transition
                // into the real location
                XtcData::Xtc& trXtc = transitionXtc();
                trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
                PGPEvent* pgpEvent = &m_pool->pgpEvents[index];
                pgpEvent->transitionDgram = trDgram;

                if (service == XtcData::TransitionId::Enable) {
                    m_running = true;
                }
                else if (service == XtcData::TransitionId::Disable) { // Sweep out L1As
                    m_running = false;
                    logging::debug("Sweeping out L1Accepts and SlowUpdates");
                    _timeout(TimeMax);
                }

                _sendToTeb(*dgram, index);
            }
        }
    }
    logging::info("Worker thread finished");
}

#include <unistd.h>

void UdpEncoder::_udpReceiver()
{
    logging::info("UDP receiver thread started");

    fd_set readfds, masterfds;
    struct timeval timeout;

    FD_ZERO(&masterfds);
    FD_SET(_dataFd, &masterfds);

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            // shutting down
            break;
        }

        memcpy(&readfds, &masterfds, sizeof(fd_set));
        timeout.tv_sec = 10;
        timeout.tv_usec = 0;
        if (select(_dataFd+1, &readfds, NULL, NULL, &timeout) < 0) {
            logging::error("select: error");
            break;
        }
        if (m_terminate.load(std::memory_order_relaxed)) {
            // shutting down
            break;
        }
        if (FD_ISSET(_dataFd, &readfds)) {
            logging::debug("%s read FD is set", __PRETTY_FUNCTION__);
            process();
        }
    }
    logging::info("UDP receiver thread finished");
}

void UdpEncoder::setMissingData(std::string errMsg)
{
    if (!m_missingData) {
        m_missingData = true;
        logging::critical("%s", errMsg.c_str());
        json msg = createAsyncErrMsg(m_para->alias, errMsg);
        m_notifySocket.send(msg.dump());
    }
}

void UdpEncoder::setOutOfOrder(std::string errMsg)
{
    if (!m_outOfOrder) {
        m_outOfOrder = true;
        logging::critical("%s", errMsg.c_str());
        json msg = createAsyncErrMsg(m_para->alias, errMsg);
        m_notifySocket.send(msg.dump());
    }
}

void UdpEncoder::process()
{
    // Protect against namesLookup not being stable before Enable
    if (m_running.load(std::memory_order_relaxed)) {
        ++m_nUpdates;
        logging::debug("%s process", m_udpMonitor->name().c_str());

        XtcData::Dgram* dgram;
        if (m_bufferFreelist.try_pop(dgram)) { // If a buffer is available...

            dgram->xtc = {{XtcData::TypeId::Parent, 0}, {nodeId}};

            const void* bufEnd = (char*)dgram + m_pool->pebble.bufferSize();
            event(*dgram, bufEnd, nullptr);    // PGPEvent not needed in this case

            m_pvQueue.push(dgram);
        }
        else {
            logging::error("%s: buffer not available, frame dropped", __PRETTY_FUNCTION__);
            ++m_nMissed;                       // Else count it as missed
            (void) _junkFrame();
        }
    } else {
        logging::debug("%s: m_running is false, frame dropped", __PRETTY_FUNCTION__);
        (void) _junkFrame();
    }
}

int UdpEncoder::_readFrame(encoder_frame_t *frame, bool& missing)
{
    int rv = 0;
    ssize_t recvlen;

    // peek data
    recvlen = recvfrom(_dataFd, frame, sizeof(encoder_frame_t), MSG_DONTWAIT | MSG_PEEK, 0, 0);
    // check length
    if (recvlen != (ssize_t) sizeof(encoder_frame_t)) {
        if (recvlen == -1) {
            perror("recvfrom(MSG_PEEK)");
        }
        logging::error("received UDP length %zd, expected %zd", recvlen, sizeof(encoder_frame_t));
        // TODO discard frame of the wrong size
    } else {
        // byte swap
        frame->header.frameCount = ntohs(frame->header.frameCount);
    }
    if (m_resetHwCount == false) {
        uint16_t expect16 = (uint16_t)(1 + m_count + m_countOffset);
        if (frame->header.frameCount != expect16) {
            // frame count doesn't match
            logging::debug("recvfrom(MSG_PEEK) frameCount %hu (expected %hu)\n", frame->header.frameCount, expect16);
            // trigger MissingData damage
            missing = true;
            // return empty frame with expected frame count
            bzero(frame, sizeof(encoder_frame_t));
            frame->header.frameCount = htons(expect16);
            frame->header.majorVersion = htons(UdpEncoder::MajorVersion);
            frame->header.minorVersion = UdpEncoder::MinorVersion;
            frame->header.microVersion = UdpEncoder::MicroVersion;
            return (0);
        }
    }

    // read data
    recvlen = recvfrom(_dataFd, frame, sizeof(encoder_frame_t), MSG_DONTWAIT, 0, 0);
    // check length
    if (recvlen != (ssize_t) sizeof(encoder_frame_t)) {
        if (recvlen == -1) {
            perror("recvfrom");
        }
        logging::error("received UDP length %zd, expected %zd", recvlen, sizeof(encoder_frame_t));
        rv = 1; // error
    } else {
        // byte swap
        frame->header.frameCount = ntohs(frame->header.frameCount);
        frame->header.majorVersion = ntohs(frame->header.majorVersion);
        frame->channel[0].encoderValue = ntohl(frame->channel[0].encoderValue);
        frame->channel[0].timing = ntohl(frame->channel[0].timing);
        frame->channel[0].scale = ntohs(frame->channel[0].scale);
        frame->channel[0].scaleDenom = ntohs(frame->channel[0].scaleDenom);

        logging::debug("     frameCount    %-7u", frame->header.frameCount);
        logging::debug("     version       %u.%u.%u", frame->header.majorVersion,
                                                      frame->header.minorVersion,
                                                      frame->header.microVersion);
        char buf[16];
        snprintf(buf, sizeof(buf), "%s", frame->header.hardwareID);
        logging::debug("     hardwareID    \"%s\"",  buf);
        logging::debug("ch0  encoderValue  %7u", frame->channel[0].encoderValue);
        logging::debug("ch0  timing        %7u", frame->channel[0].timing);
        logging::debug("ch0  scale         %7u", (unsigned)frame->channel[0].scale);
        logging::debug("ch0  scaleDenom    %7u", (unsigned)frame->channel[0].scaleDenom);
        logging::debug("ch0  error         %7u", (unsigned)frame->channel[0].error);
        logging::debug("ch0  mode          %7u", (unsigned)frame->channel[0].mode);
    }
    return (rv);
}

int UdpEncoder::_junkFrame()
{
    int rv = 0;
    ssize_t recvlen;
    encoder_frame_t junk;

    // read data
    recvlen = recvfrom(_dataFd, &junk, sizeof(junk), MSG_DONTWAIT, 0, 0);
    // check length
    if (recvlen != (ssize_t) sizeof(encoder_frame_t)) {
        if (recvlen == -1) {
            perror("recvfrom");
        }
        logging::error("%s: received length %zd, expected %zd",
                       __PRETTY_FUNCTION__, recvlen, sizeof(junk));
        rv = 1; // error
    }
    return (rv);
}

void UdpEncoder::_matchUp()
{
    while (true) {
        XtcData::Dgram* pvDg;
        if (!m_pvQueue.peek(pvDg))  break;

        uint32_t pgpIdx;
        if (!m_pgpQueue.peek(pgpIdx))  break;

        Pds::EbDgram* pgpDg = reinterpret_cast<Pds::EbDgram*>(m_pool->pebble[pgpIdx]);

        _handleMatch  (*pvDg, *pgpDg);
    }
    //printf("\n");
}

void UdpEncoder::_handleMatch(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg)
{
    uint32_t pgpIdx;
    m_pgpQueue.try_pop(pgpIdx);         // Actually consume the element

    XtcData::Dgram* dgram;
    if (pgpDg.service() == XtcData::TransitionId::L1Accept) {
        pgpDg.xtc.damage.increase(pvDg.xtc.damage.value());
        auto bufEnd  = (char*)&pgpDg + m_pool->pebble.bufferSize();
        auto payload = pgpDg.xtc.alloc(pvDg.xtc.sizeofPayload(), bufEnd);
        memcpy(payload, (const void*)pvDg.xtc.payload(), pvDg.xtc.sizeofPayload());

        m_pvQueue.try_pop(dgram);       // Actually consume the element
        m_bufferFreelist.push(dgram);   // Return buffer to freelist

        ++m_nMatch;
    }
    else { // SlowUpdate
        // Allocate a transition dgram from the pool and initialize its header
        Pds::EbDgram* trDg = m_pool->allocateTr();
        *trDg = pgpDg;                  // Initialized Xtc, possibly w/ damage
        PGPEvent* pgpEvent = &m_pool->pgpEvents[pgpIdx];
        pgpEvent->transitionDgram = trDg;

        if (tsMatchDegree == 2) {       // Keep PV for the next L1A
          m_pvQueue.try_pop(dgram);     // Actually consume the element
          m_bufferFreelist.push(dgram); // Return buffer to freelist
        }
    }

    _sendToTeb(pgpDg, pgpIdx);
}

void UdpEncoder::_timeout(const XtcData::TimeStamp& timestamp)
{
    while (true) {
        uint32_t index;
        if (!m_pgpQueue.peek(index)) {
            break;
        }

        Pds::EbDgram& dgram = *reinterpret_cast<Pds::EbDgram*>(m_pool->pebble[index]);
        if (dgram.time > timestamp) {
            break;                  // dgram is newer than the timeout timestamp
        }

        uint32_t idx;
        m_pgpQueue.try_pop(idx);        // Actually consume the element
        assert(idx == index);

        if (dgram.service() == XtcData::TransitionId::L1Accept) {
          // No UDP data so mark event as damaged
          dgram.xtc.damage.increase(XtcData::Damage::TimedOut);
          ++m_nTimedOut;
          //printf("TO: %u.%09u, PGP: %u.%09u, PGP - TO: %10ld ns, svc %2d  Timeout\n",
          //       timestamp.seconds(), timestamp.nanoseconds(),
          //       dgram.time.seconds(), dgram.time.nanoseconds(),
          //       dgram.time.to_ns() - timestamp.to_ns(),
          //       dgram.service());
          logging::debug("Event timed out!! "
                         "TimeStamps: timeout %u.%09u > PGP %u.%09u [0x%08x%04x.%05x > 0x%08x%04x.%05x]",
                         timestamp.seconds(), timestamp.nanoseconds(),
                         dgram.time.seconds(), dgram.time.nanoseconds(),
                         timestamp.seconds(), (timestamp.nanoseconds()>>16)&0xfffe, timestamp.nanoseconds()&0x1ffff,
                         dgram.time.seconds(), (dgram.time.nanoseconds()>>16)&0xfffe, dgram.time.nanoseconds()&0x1ffff);
        }
        else { // SlowUpdate
            // Allocate a transition dgram from the pool and initialize its header
            Pds::EbDgram* trDg = m_pool->allocateTr();
            *trDg = dgram;              // Initialized Xtc, possibly w/ damage
            PGPEvent* pgpEvent = &m_pool->pgpEvents[index];
            pgpEvent->transitionDgram = trDg;
        }

        _sendToTeb(dgram, index);
    }
}

void UdpEncoder::_sendToTeb(const Pds::EbDgram& dgram, uint32_t index)
{
    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = (dgram.service() == XtcData::TransitionId::L1Accept)
                         ? m_pool->pebble.bufferSize()
                         : m_para->maxTrSize;
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
            triggerPrimitive->event(*m_pool, index, dgram.xtc, l3InpDg->xtc, bufEnd); // Produce
        }
    }
    m_drp.tebContributor().process(l3InpDg);
}

int UdpEncoder::drainDataFd()
{
  int rv = 0;
  unsigned count = 0;
  encoder_frame_t junk;

  if (_dataFd > 0) {
    while ((rv = recvfrom(_dataFd, &junk, sizeof(junk), MSG_DONTWAIT, 0, 0)) > 0) {
      if (rv == -1) {
        perror("recvfrom");
        break;
      }
      ++ count;
    }
    if (count > 0) {
      logging::warning("%s: drained %u frames\n", __PRETTY_FUNCTION__, count);
    }
  }

  return (rv);
}

int UdpEncoder::reset()
{
  int rv = -1;  // ERROR

  if (_dataFd > 0) {
    // drain input buffers
    rv = drainDataFd();
  }
  return (rv);
}

UdpApp::UdpApp(Parameters& para, std::shared_ptr<UdpMonitor> udpMonitor) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_udpDetector(std::make_unique<UdpEncoder>(m_para, udpMonitor, m_drp)),
    m_det(m_udpDetector.get()),
    m_unconfigure(false)
{
    Py_Initialize();                    // for use by configuration

    if (m_det == nullptr) {
        logging::critical("Error !! Could not create Detector object for %s", m_para.detType.c_str());
        throw "Could not create Detector object for " + m_para.detType;
    }
    logging::info("Ready for transitions");
}

UdpApp::~UdpApp()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    handleReset(json({}));

    Py_Finalize();                      // for use by configuration
}

void UdpApp::_disconnect()
{
    m_drp.disconnect();
    m_det->shutdown();
}

void UdpApp::_unconfigure()
{
    m_drp.pool.shutdown();  // Release Tr buffer pool
    m_drp.unconfigure();    // TebContributor must be shut down before the worker
    m_udpDetector->unconfigure();
    m_unconfigure = false;
}

json UdpApp::connectionInfo()
{
    std::string ip = m_para.kwargs.find("ep_domain") != m_para.kwargs.end()
                   ? getNicIp(m_para.kwargs["ep_domain"])
                   : getNicIp(m_para.kwargs["forceEnet"] == "yes");
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo(ip);
    body["connect_info"].update(bufInfo);
    return body;
}

void UdpApp::connectionShutdown()
{
    m_drp.shutdown();
}

void UdpApp::_error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void UdpApp::handleConnect(const nlohmann::json& msg)
{
    m_det->nodeId = msg["body"]["drp"][std::to_string(getId())]["drp_id"];
    m_det->connect(msg, std::to_string(getId()));

    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect");
        logging::error("%s", errorMsg.c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void UdpApp::handleDisconnect(const json& msg)
{
    // Carry out the queued Unconfigure, if there was one
    if (m_unconfigure) {
        _unconfigure();
    }

    _disconnect();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void UdpApp::handlePhase1(const json& msg)
{
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in UdpEncoderApp", key.c_str());

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

        std::string config_alias = msg["body"]["config_alias"];
        unsigned error = m_det->configure(config_alias, xtc, bufEnd);
        if (error) {
            std::string errorMsg = "Failed transition phase 1";
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }

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
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        m_drp.runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp.endrun(phase1Info);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
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
        m_udpDetector->reset(); // needed?
        logging::debug("handlePhase1 enable complete");
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void UdpApp::handleReset(const nlohmann::json& msg)
{
    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    _unconfigure();
    _disconnect();
    connectionShutdown();
}

} // namespace Drp

int createUdpSocket(int port)
{
  struct sockaddr_in myaddr; /* our address */
  int fd; /* our socket */

  /* create a UDP socket */
  if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("socket");
    return 0;
  }

  /* bind the socket to any valid IP address and a specific port */
  memset((char *)&myaddr, 0, sizeof(myaddr));
  myaddr.sin_family = AF_INET;
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(port);
  if (bind(fd, (struct sockaddr *)&myaddr, sizeof(myaddr)) < 0) {
    perror("bind");
    return 0;
  }
  /* set receive buffer size */
  if (setrcvbuf(fd, UDP_RCVBUF_SIZE) < 0) {
    printf("Error: Failed to set socket receive buffer to %u bytes\n\r", UDP_RCVBUF_SIZE);
    return 0;
  }
  return (fd);
}

int setrcvbuf(int socketFd, unsigned size)
{
  if (::setsockopt(socketFd, SOL_SOCKET, SO_RCVBUF,
       (char*)&size, sizeof(size)) < 0) {
    perror("setsockopt");
    return -1;
  }
  return 0;
}

int main(int argc, char* argv[])
{
    Drp::Parameters para;
    std::string kwargs_str;
    int c;
    while((c = getopt(argc, argv, "p:L:o:l:D:S:C:d:u:k:P:M:01v")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.outputDir = optarg;
                break;
            case 'l':
                para.laneMask = std::stoul(optarg, nullptr, 16);
                break;
            case 'D':
                para.detType = optarg;  // Defaults to 'pv'
                break;
            case 'S':
                para.serNo = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'k':
                kwargs_str = kwargs_str.empty()
                           ? optarg
                           : kwargs_str + ", " + optarg;
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            //  Indicate level of timestamp matching (ugh)
            case '0':
                tsMatchDegree = 0;
                break;
            case '1':
                tsMatchDegree = 1;
                break;
            case 'v':
                ++para.verbose;
                break;
            case 'L':
                para.loopbackPort = std::stoi(optarg);
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

    // Allow detType to be overridden, but generally, psana expects 'encoder'
    if (para.detType.empty()) {
      para.detType = "encoder";
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        return 1;
    }
    para.detName = para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));

    para.maxTrSize = 256 * 1024;
    try {
        get_kwargs(kwargs_str, para.kwargs);
        for (const auto& kwargs : para.kwargs) {
            if (kwargs.first == "forceEnet")     continue;
            if (kwargs.first == "ep_fabric")     continue;
            if (kwargs.first == "ep_domain")     continue;
            if (kwargs.first == "ep_provider")   continue;
            if (kwargs.first == "sim_length")    continue;  // XpmDetector
            if (kwargs.first == "timebase")      continue;  // XpmDetector
            if (kwargs.first == "pebbleBufSize") continue;  // DrpBase
            if (kwargs.first == "batching")      continue;  // DrpBase
            if (kwargs.first == "directIO")      continue;  // DrpBase
            if (kwargs.first == "match_tmo_ms")  continue;
            logging::critical("Unrecognized kwarg '%s=%s'\n",
                              kwargs.first.c_str(), kwargs.second.c_str());
            return 1;
        }
        Drp::UdpApp app(para, std::make_shared<Drp::UdpMonitor>(para));
        app.run();
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
}

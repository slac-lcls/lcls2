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
#include <cmath>
#include <Python.h>
#include <arpa/inet.h>
#include <sys/prctl.h>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "RunInfoDef.hh"
#include "xtcdata/xtc/Damage.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/fast_monotonic_clock.hh"


#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

#define MAX_ENC_VALUES 2
#define POLYNOMIAL_ORDER 1

using namespace XtcData;
using namespace Pds;
using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

// forward declarations
int setrcvbuf(int socketFd, unsigned size);
int createUdpSocket(int port);

class EncoderDef:public VarDef
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
      error
    };

  EncoderDef()
  {
      NameVec.push_back({"encoderValue", Name::UINT32});
      // frameCount is common to all channels
      NameVec.push_back({"frameCount", Name::UINT16});
      NameVec.push_back({"timing", Name::UINT32});
      NameVec.push_back({"scale", Name::UINT16});
      NameVec.push_back({"scaleDenom", Name::UINT16});
      NameVec.push_back({"mode", Name::UINT8});
      NameVec.push_back({"error", Name::UINT8});
   }
} RawDef, InterpolatedDef;


namespace Drp {

struct UdpParameters : public Parameters
{
    unsigned slowGroup;
    bool interpolating;
};

UdpReceiver::UdpReceiver(const UdpParameters& para,
                         size_t               nBuffers) :
    m_para          (para),
    m_encQueue      (nBuffers),
    m_bufferFreelist(m_encQueue.size()),
    m_terminate     (false),
    m_outOfOrder    (false),
    m_missingData   (false),
    m_notifySocket  {&m_context, ZMQ_PUSH},
    m_nUpdates      (0),
    m_nMissed       (0)
{
    // ZMQ socket for reporting errors
    m_notifySocket.connect({"tcp://" + m_para.collectionHost + ":" + std::to_string(CollectionApp::zmq_base_port + m_para.partition)});

    // UDP socket for receiving data
    int dataPort = (m_para.loopbackPort) ? m_para.loopbackPort : UdpEncoder::DefaultDataPort;
    _dataFd = createUdpSocket(dataPort);
    logging::debug("createUdpSocket(%d) returned %d", dataPort, _dataFd);
}

UdpReceiver::~UdpReceiver()
{
    if (_dataFd > 0) {
        close(_dataFd);
    }
}

void UdpReceiver::start()
{
    m_resetHwCount = true;

    m_encQueue.startup();
    m_bufferFreelist.startup();
    size_t bufSize = sizeof(Dgram) + sizeof(encoder_frame_t);
    m_buffer.resize(m_encQueue.size() * bufSize);
    for(unsigned i = 0; i < m_encQueue.size(); ++i) {
        m_bufferFreelist.push(reinterpret_cast<Dgram*>(&m_buffer[i * bufSize]));
    }

    m_terminate.store(false, std::memory_order_release);

    m_udpReceiverThread = std::thread{&UdpReceiver::_udpReceiver, this};

    if (m_para.loopbackPort) {
        _loopbackInit();        // LOOPBACK TEST
    }

    logging::info("%s started", name().c_str());
}

void UdpReceiver::stop()
{
    m_terminate.store(true, std::memory_order_release);

    if (m_udpReceiverThread.joinable()) {
        m_udpReceiverThread.join();
    }

    m_encQueue.shutdown();
    m_bufferFreelist.shutdown();

    logging::info("%s stopped", name().c_str());
}

void UdpReceiver::_loopbackInit()
{
    logging::debug("%s (port = %d)", __PRETTY_FUNCTION__, m_para.loopbackPort);

    if (m_para.loopbackPort > 0) {
        m_loopbackFd = socket(AF_INET,SOCK_DGRAM, 0);
        if (m_loopbackFd == -1) {
            perror("socket");
            logging::error("failed to create loopback socket");
        }

        bzero(&m_loopbackAddr, sizeof(m_loopbackAddr));
        m_loopbackAddr.sin_family = AF_INET;
        m_loopbackAddr.sin_addr.s_addr=inet_addr("127.0.0.1");
        m_loopbackAddr.sin_port=htons(m_para.loopbackPort);

        m_loopbackFrameCount = 0;
    }
}

void UdpReceiver::_loopbackFini()
{
    logging::debug("%s", __PRETTY_FUNCTION__);

    if (m_loopbackFd > 0) {
        if (close(m_loopbackFd)) {
            logging::error("failed to close loopback socket");
        }
    }
}

void UdpReceiver::loopbackSend()
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

void UdpReceiver::_udpReceiver()
{
    logging::info("UDP receiver thread is starting with process ID %lu", syscall(SYS_gettid));
    if (prctl(PR_SET_NAME, "drp_udp/Receiver", 0, 0, 0) == -1) {
        perror("prctl");
    }

    fd_set readfds, masterfds;
    struct timeval timeout;

    FD_ZERO(&masterfds);
    FD_SET(_dataFd, &masterfds);

    m_nUpdates = 0;
    m_nMissed  = 0;

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

void UdpReceiver::setMissingData(std::string errMsg)
{
    if (!m_missingData) {
        m_missingData = true;
        logging::critical("%s", errMsg.c_str());
        json msg = createAsyncErrMsg(m_para.alias, errMsg);
        m_notifySocket.send(msg.dump());
    }
}

void UdpReceiver::setOutOfOrder(std::string errMsg)
{
    if (!m_outOfOrder) {
        m_outOfOrder = true;
        logging::critical("%s", errMsg.c_str());
        json msg = createAsyncErrMsg(m_para.alias, errMsg);
        m_notifySocket.send(msg.dump());
    }
}

void UdpReceiver::_read(Dgram& dgram)
{
    const void* bufEnd = dgram.xtc.payload() + sizeof(encoder_frame_t);
    encoder_frame_t* frame = (encoder_frame_t*)dgram.xtc.alloc(sizeof(encoder_frame_t), bufEnd);
    bool missing = false;

    // read from the udp socket that triggered select()
    int rv = _readFrame(frame, missing);

    // if reading frame failed, record damage and return early
    if (rv) {
        dgram.xtc.damage.increase(Damage::UserDefined);
        logging::critical("%s: failed to read UDP frame", __PRETTY_FUNCTION__);
        return;
    }

    // if missing data reported, record damage
    if (missing) {
        // record damage
        dgram.xtc.damage.increase(Damage::MissingData);
        // report first occurance
        if (!getMissingData()) {
            char errmsg[128];
            snprintf(errmsg, sizeof(errmsg),
                     "Missing data for frame %hu", frame->header.frameCount);
            setMissingData(errmsg);
        }
    }

    logging::debug("%s: frame=%hu  encoderValue=%u  timing=%u  scale=%u  scaleDenom=%u  mode=%u  error=%u  version=%u.%u.%u",
                   __PRETTY_FUNCTION__,
                   frame->header.frameCount,
                   frame->channel[0].encoderValue,
                   frame->channel[0].timing,
                   (unsigned) frame->channel[0].scale,
                   (unsigned) frame->channel[0].scaleDenom,
                   (unsigned) frame->channel[0].mode,
                   (unsigned) frame->channel[0].error,
                   (unsigned) frame->header.majorVersion,
                   (unsigned) frame->header.minorVersion,
                   (unsigned) frame->header.microVersion);

    // reset frame counter
    if (m_resetHwCount) {
        m_count = 0;
        m_countOffset = frame->header.frameCount - 1;
        m_resetHwCount = false;
    }

    // update frame counter
    uint16_t stuck16 = (uint16_t)(m_count + m_countOffset);
    ++m_count;
    uint16_t sum16 = (uint16_t)(m_count + m_countOffset);

    if (!getOutOfOrder()) {
        char errmsg[128];
        // check for out-of-order condition
        if (frame->header.frameCount == stuck16) {
            snprintf(errmsg, sizeof(errmsg),
                     "Out-of-order: frame count %hu repeated in consecutive frames", stuck16);
            setOutOfOrder(errmsg);

        } else if (frame->header.frameCount != sum16) {
            snprintf(errmsg, sizeof(errmsg),
                     "Out-of-order: hw count (%hu) != sw count (%hu) + offset (%u) == (%hu)",
                     frame->header.frameCount, m_count, m_countOffset, sum16);
            setOutOfOrder(errmsg);
        }
    }

    // record damage
    if (m_outOfOrder) {
        dgram.xtc.damage.increase(Damage::OutOfOrder);
    }
}

void UdpReceiver::process()
{
    ++m_nUpdates;
    logging::debug("%s process", name().c_str());

    Dgram* dgram;
    if (m_bufferFreelist.try_pop(dgram)) { // If a buffer is available...

        dgram->xtc = {{TypeId::Parent, 0}, {0}};

        _read(*dgram);                     // read the frame into the Dgram

        m_encQueue.push(dgram);
    }
    else {
        logging::error("%s: buffer not available, frame dropped", __PRETTY_FUNCTION__);
        ++m_nMissed;                       // Else count it as missed
        (void) _junkFrame();
    }
}

int UdpReceiver::_readFrame(encoder_frame_t *frame, bool& missing)
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

int UdpReceiver::_junkFrame()
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

bool UdpReceiver::consume()
{
    Dgram* dgram;
    auto rc = m_encQueue.try_pop(dgram);          // Actually consume the element
    if (rc)   m_bufferFreelist.push(dgram);       // Return buffer to freelist

    return rc;
}

void UdpReceiver::drain()
{
    while (consume()) {                            // Drain stale encoder buffers
        printf("*** Drained stale encoder buffer: sz %u, %u\n",
               m_encQueue.guess_size(), m_bufferFreelist.guess_size());
    }
}

int UdpReceiver::drainDataFd()
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

int UdpReceiver::reset()
{
  int rv = -1;  // ERROR

  if (_dataFd > 0) {
    // drain input buffers
    rv = drainDataFd();
  }
  return (rv);
}


Pgp::Pgp(const UdpParameters& para, MemPool& pool, Detector* det) :
    PgpReader(para, pool, MAX_RET_CNT_C, 32),
    m_det(det),
    m_available(0), m_current(0), m_nDmaRet(0)
{
    if (pool.setMaskBytes(para.laneMask, det->virtChan)) {
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
        if (m_available == 0) {
            return nullptr;
        }
    }

    EbDgram* dgram = _handle(evtIndex);
    m_current++;
    return dgram;
}


static void _polyfit(unsigned i0,
                     const std::vector<double> &t,
                     const std::vector<double> &v,
                     std::vector<double> &coeff)
{
    // Copy values into a time-ordered vector
    std::vector<double> val(v.size());
    for(unsigned i = 0; i < v.size(); i++) {
        unsigned j = (i0 + i) % v.size();
        val[i] = v[j];
    }

    // Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
    Eigen::MatrixXd T(t.size(), coeff.size());
    Eigen::VectorXd V = Eigen::VectorXd::Map(&val.front(), val.size());
    Eigen::VectorXd result;

    // Populate the matrix
    double t0 = t[i0 % t.size()];
    for(size_t i = 0 ; i < t.size(); ++i) {
        size_t k = (i0 + i) % t.size();
        for(size_t j = 0; j < coeff.size(); ++j) {
            T(i, j) = pow(t.at(k) - t0, j);
        }
    }
    //std::cout<<T<<std::endl;

    // Solve for linear least square fit
    result = T.householderQr().solve(V);
    for (unsigned k = 0; k < coeff.size(); k++) {
        coeff[k] = result[k];
    }
}

void Interpolator::update(TimeStamp t, unsigned v)
{
    auto idx = _idx++;
    _idx %= _t.size();

    logging::debug("Interpolator::update: idx      %d,    t %u.%09u, v %u", idx, t.seconds(), t.nanoseconds(), v);

    _t[idx] = t.asDouble();
    _v[idx] = double(v);

    if (_t[_t.size() - 1] != 0.0) { // True when MAX_ENC_VALUES or more points were collected
        _polyfit(_idx, _t, _v, _coeff);
    }
}

unsigned Interpolator::calculate(TimeStamp ts, Damage& damage) const
{
    unsigned v;
    if (_t[_t.size() - 1] != 0.0) { // True when arrays are full
        // Sum up the terms of the polynomial
        double val  = 0;
        double tPwr = 1;
        double t0   = _t[_idx % _t.size()];
        double t    = ts.asDouble() - t0;
        for(unsigned i = 0; i < _coeff.size(); ++i) {
            val  += _coeff[i] * tPwr;
            tPwr *= t;
        }
        v = unsigned(std::round(val));
    } else {
        v = unsigned(_v[_idx - 1]);   // Return the most recent value available
        damage.increase(Damage::MissingData);
    }

    logging::debug("Interpolator::calc:   idx %d, t %u.%09u, v %u", _idx, ts.seconds(), ts.nanoseconds(), v);

    return v;
}


UdpEncoder::UdpEncoder(UdpParameters& para, MemPoolCpu& pool) :
    XpmDetector(&para, &pool)
{
    virtChan = 0;
}

unsigned UdpEncoder::connect(const json& msg, const std::string& id, std::string& errorMsg)
{
    // First call the parent class connect() with the required JSON and id.
    XpmDetector::connect(msg, id);

    // Create the UdpReceiver for this detector.
    try {
        const auto& para = *static_cast<UdpParameters*>(m_para);
        m_udpReceiver = std::make_shared<UdpReceiver>(para, m_pool->nbuffers());
    }
    catch(const std::string& error) {
        logging::error("Failed to create UdpReceiver: %s", error.c_str());
        m_udpReceiver.reset();
        errorMsg = error;
        return 1;
    }
    return 0;
}

unsigned UdpEncoder::disconnect()
{
    XpmDetector::shutdown();
    m_udpReceiver.reset();
    return 0;
}

void UdpEncoder::addNames(unsigned segment, Xtc& xtc, const void* bufEnd)
{
    // raw
    Alg encoderRawAlg("raw", UdpEncoder::MajorVersion, UdpEncoder::MinorVersion, UdpEncoder::MicroVersion);
    NamesId rawNamesId(nodeId, RawNamesIndex);
    Names&  rawNames = *new(xtc, bufEnd) Names(bufEnd,
                                               m_para->detName.c_str(), encoderRawAlg,
                                               m_para->detType.c_str(), m_para->serNo.c_str(), rawNamesId, segment);
    rawNames.add(xtc, bufEnd, RawDef);
    m_namesLookup[rawNamesId] = NameIndex(rawNames);

    if (static_cast<UdpParameters*>(m_para)->interpolating) {
        // interpolated
        Alg encoderInterpolatedAlg("interpolated", UdpEncoder::MajorVersion, UdpEncoder::MinorVersion, UdpEncoder::MicroVersion);
        NamesId interpolatedNamesId(nodeId, InterpolatedNamesIndex);
        Names&  interpolatedNames = *new(xtc, bufEnd) Names(bufEnd,
                                                            m_para->detName.c_str(), encoderInterpolatedAlg,
                                                            m_para->detType.c_str(), m_para->serNo.c_str(), interpolatedNamesId, segment);
        interpolatedNames.add(xtc, bufEnd, InterpolatedDef);
        m_namesLookup[interpolatedNamesId] = NameIndex(interpolatedNames);
    }
}

  //std::string UdpEncoder::sconfigure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
unsigned UdpEncoder::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
    logging::debug("entered %s", __PRETTY_FUNCTION__);

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    addNames(0, xtc, bufEnd);

    return 0;
}

unsigned UdpEncoder::unconfigure()
{
    m_namesLookup.clear();   // erase all elements

    return 0;
}

// only doing the CreateData for the “raw” case on events where the encoder is read out.
void UdpEncoder::event_(Dgram& dgram, const void* const bufEnd, const encoder_frame_t& frame,
                        uint32_t *rawValue, uint32_t *interpolatedValue)
{
    // ----- CreateData  ------------------------------------------------------

    if (interpolatedValue != nullptr) {
        // interpolated
        NamesId namesId1(nodeId, InterpolatedNamesIndex);
        CreateData interpolated(dgram.xtc, bufEnd, m_namesLookup, namesId1);

        // ...encoderValue
        interpolated.set_value(EncoderDef::encoderValue, *interpolatedValue);

        // ...frameCount
        interpolated.set_value(EncoderDef::frameCount, frame.header.frameCount);

        // ...timing
        interpolated.set_value(EncoderDef::timing, frame.channel[0].timing);

        // ...scale
        interpolated.set_value(EncoderDef::scale, frame.channel[0].scale);

        // ...scaleDenom
        interpolated.set_value(EncoderDef::scaleDenom, frame.channel[0].scaleDenom);

        // ...mode
        interpolated.set_value(EncoderDef::mode, frame.channel[0].mode);

        // ...error
        interpolated.set_value(EncoderDef::error, frame.channel[0].error);
    }

    if (rawValue != nullptr) {
        // raw
        NamesId namesId2(nodeId, RawNamesIndex);
        CreateData raw(dgram.xtc, bufEnd, m_namesLookup, namesId2);

        // ...encoderValue
        raw.set_value(EncoderDef::encoderValue, *rawValue);

        // ...frameCount
        raw.set_value(EncoderDef::frameCount, frame.header.frameCount);

        // ...timing
        raw.set_value(EncoderDef::timing, frame.channel[0].timing);

        // ...scale
        raw.set_value(EncoderDef::scale, frame.channel[0].scale);

        // ...scaleDenom
        raw.set_value(EncoderDef::scaleDenom, frame.channel[0].scaleDenom);

        // ...mode
        raw.set_value(EncoderDef::mode, frame.channel[0].mode);

        // ...error
        raw.set_value(EncoderDef::error, frame.channel[0].error);
    }
}


UdpDrp::UdpDrp(UdpParameters& para, MemPoolCpu& pool, UdpEncoder& det, ZmqContext& context) :
    DrpBase       (para, pool, det, context),
    m_para        (para),
    m_det         (det),
    m_pgp         (para, pool, &det),
    m_interpolator(MAX_ENC_VALUES, POLYNOMIAL_ORDER),
    m_evtQueue    (pool.nbuffers()),
    m_terminate   (false),
    m_running     (false)
{
    if (para.kwargs.find("slowGroup") != para.kwargs.end())
        para.slowGroup = std::stoi(para.kwargs["slowGroup"]);
    else
        para.slowGroup = -1;   // default: interpolation disabled

    logging::debug("%s: slowGroup = %u", __PRETTY_FUNCTION__, para.slowGroup);
    if ((m_para.slowGroup < 0) || (m_para.slowGroup > 7)) {   // valid readout groups are 0-7
        para.interpolating = false;
        logging::info("Interpolation disabled");
    } else {
        para.interpolating = true;
        logging::info("Interpolation enabled");
    }
}

std::string UdpDrp::connect(const json& msg, size_t id)
{
  std::string errorMsg = DrpBase::connect(msg, id);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    // Configure interpolation based on slowGroup.
    if (m_para.kwargs.find("encTprAlias") != m_para.kwargs.end()) {
      std::string encTprAlias = const_cast<UdpParameters&>(m_para).kwargs["encTprAlias"];
        for (auto it : msg["body"]["tpr"].items()) {
            const_cast<UdpParameters&>(m_para).interpolating = true;
            const_cast<UdpParameters&>(m_para).slowGroup = it.value()["det_info"]["readout"];
            logging::info("Interpolation enabled using group %u", m_para.slowGroup);
            if (it.value()["proc_info"]["alias"] == encTprAlias)
                break;
        }
    }

    return std::string();
}

std::string UdpDrp::configure(const json& msg)
{
    std::string errorMsg = DrpBase::configure(msg);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    m_workerThread = std::thread{&UdpDrp::_worker, this};

    return std::string();
}

unsigned UdpDrp::unconfigure()
{
    DrpBase::unconfigure(); // TebContributor must be shut down before the worker

    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }

    return 0;
}

int UdpDrp::_setupMetrics(const std::shared_ptr<MetricExporter> exporter)
{
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"detseg", std::to_string(m_para.detSegment)},
                                              {"alias", m_para.alias}};
    m_nEvents = 0;
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return m_nEvents;});
    exporter->add("drp_update_rate", labels, MetricType::Rate,
                  [&](){return m_det.udpReceiver() ? m_det.udpReceiver()->nUpdates() : 0;});
    m_nMatch = 0;
    exporter->add("drp_match_count", labels, MetricType::Counter,
                  [&](){return m_nMatch;});
    exporter->add("drp_miss_count", labels, MetricType::Counter,
                  [&](){return m_det.udpReceiver() ? m_det.udpReceiver()->nMissed() : 0;});
    m_nTimedOut = 0;
    exporter->add("drp_timeout_count", labels, MetricType::Counter,
                  [&](){return m_nTimedOut;});

    exporter->add("drp_worker_input_queue", labels, MetricType::Gauge,
                  [&](){return m_evtQueue.guess_size();});
    exporter->constant("drp_worker_queue_depth", labels, m_evtQueue.size());

    // Borrow this for awhile
    exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
                  [&](){return m_det.udpReceiver()->encQueue().guess_size();});

    exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                  [&](){return m_pgp.nDmaRet();});
    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return m_pgp.dmaBytes();});
    exporter->add("drp_dma_size", labels, MetricType::Gauge,
                  [&](){return m_pgp.dmaSize();});
    exporter->add("drp_th_latency", labels, MetricType::Gauge,
                  [&](){return m_pgp.latency();});
    exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                  [&](){return m_pgp.nDmaErrors();});
    exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                  [&](){return m_pgp.nNoComRoG();});
    exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                  [&](){return m_pgp.nMissingRoGs();});
    exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                  [&](){return m_pgp.nTmgHdrError();});
    exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpJumps();});
    exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                  [&](){return m_pgp.nNoTrDgrams();});

    exporter->add("drp_num_pgp_in_user", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInUser();});
    exporter->add("drp_num_pgp_in_hw", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInHw();});
    exporter->add("drp_num_pgp_in_prehw", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInPreHw();});
    exporter->add("drp_num_pgp_in_rx", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInRx();});

    return 0;
}

void UdpDrp::_worker()
{
    logging::info("Worker thread started");

    m_terminate.store(false, std::memory_order_release);

    const ms_t tmo{ m_para.kwargs.find("match_tmo_ms") != m_para.kwargs.end()             ?
                    std::stoul(const_cast<UdpParameters&>(m_para).kwargs["match_tmo_ms"]) :
                    1500 };

    const ms_t no_tmo{ 0 };

    logging::info("Worker thread is starting with process ID %lu", syscall(SYS_gettid));
    if (prctl(PR_SET_NAME, "drp_udp/Worker", 0, 0, 0) == -1) {
        perror("prctl");
    }

    // Reset counters to avoid 'jumping' errors reconfigures
    pool.resetCounters();
    m_pgp.resetEventCounter();

    // (Re)initialize the queues
    m_evtQueue.startup();
    m_det.udpReceiver()->start();

    // Set up monitoring
    auto exporter = std::make_shared<MetricExporter>();
    if (exposer()) {
        exposer()->RegisterCollectable(exporter);

        if (_setupMetrics(exporter))  return;
    }

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        EbDgram* dgram = m_pgp.next(index);
        if (dgram) {
            m_nEvents++;

            m_evtQueue.push(index);

            _process(dgram);            // Was _matchUp
        }
        else {
            // If there are any PGP datagrams stacked up, try to match them
            // up with any encoder updates that may have arrived
            _process(dgram);            // Was _matchUp

            // Time out older encoder packets and datagrams
            _timeout(m_running ? tmo : no_tmo);

            // Time out batches for the TEB
            tebContributor().timeout();
        }
    }

    // Flush the DMA buffers
    m_pgp.flush();

    m_det.udpReceiver()->stop();
    m_evtQueue.shutdown();

    if (exposer())  exporter.reset();

    logging::info("Worker thread finished");
}

void UdpDrp::_process(EbDgram* dgram)
{
    auto& encQueue = m_det.udpReceiver()->encQueue();
    while (true) {
        uint32_t evtIdx;
        if (!m_evtQueue.peek(evtIdx))  break;

        EbDgram* evtDg = reinterpret_cast<EbDgram*>(pool.pebble[evtIdx]);
        if (evtDg->service() != TransitionId::L1Accept) {
            _handleTransition(evtIdx, evtDg);
            continue;
        }

        if (m_para.interpolating) {
            assert((m_para.slowGroup >= 0) && (m_para.slowGroup <= 7));
            if (dgram && (dgram->readoutGroups() & (1 << m_para.slowGroup)) &&
                (dgram->service() == TransitionId::L1Accept)) {
                const ms_t tmo(1500);
                auto tInitial = fast_monotonic_clock::now();
                Dgram* encDg;
                while (!encQueue.peek(encDg)) {  // Wait for an encoder value
                    if (fast_monotonic_clock::now() - tInitial > tmo) {
                        logging::warning("encQueue peek timed out: %u\n", encQueue.guess_size());
                        return;
                    }
                    std::this_thread::yield();
                }
                auto& frame = *(encoder_frame_t*)(encDg->xtc.payload());
                auto  value = frame.channel[0].encoderValue;

                // Associate the current L1Accept's timestamp with the latest encoder value
                m_interpolator.update(dgram->time, value);

                // Handle all events that have accumulated on the queue
                while (true) {
                    uint32_t pgpIdx;
                    uint32_t rawValue = value;
                    if (!m_evtQueue.peek(pgpIdx))  break;

                    // Deal with intermediate SlowUpdates
                    auto pgpDg = reinterpret_cast<EbDgram*>(pool.pebble[pgpIdx]);
                    if (pgpDg->service() != TransitionId::L1Accept) {
                        _handleTransition(pgpIdx, pgpDg);
                        continue;
                    }

                    // Update the encoder value
                    uint32_t interpolatedValue = m_interpolator.calculate(pgpDg->time, pgpDg->xtc.damage);
                    if (pgpDg != dgram) {
                        _handleL1Accept(*pgpDg, frame, nullptr, &interpolatedValue);
                    } else {
                        _handleL1Accept(*pgpDg, frame, &rawValue, &interpolatedValue);
                    }

                    if (pgpDg == dgram)  return;
                }
            } else {
                //  Trigger a fast flush of the events awaiting interpolation
                if (dgram && dgram->service() == TransitionId::Disable) {
                    m_running = false;
                }
                break;
            }
        } else {
            // not interpolating
            Dgram* encDg;
            if (!encQueue.peek(encDg))  break;
            evtDg->xtc.damage.increase(encDg->xtc.damage.value());

            auto frame = (const encoder_frame_t*)(encDg->xtc.payload());
            uint32_t rawValue = frame->channel[0].encoderValue;
            _handleL1Accept(*evtDg, *frame, &rawValue, nullptr);
        }
    }
}

void UdpDrp::_handleTransition(uint32_t pebbleIdx, EbDgram* pebbleDg)
{
    // Find the transition dgram in the pool and initialize its header
    EbDgram* trDgram = pool.transitionDgrams[pebbleIdx];
    if (trDgram) {                      // nullptr happen during shutdown
        *trDgram = *pebbleDg;

        TransitionId::Value service = trDgram->service();
        if (service != TransitionId::SlowUpdate) {
            // copy the temporary xtc created on phase 1 of the transition
            // into the real location
            Xtc& trXtc = m_det.transitionXtc();
            trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
            const void* bufEnd = (char*)trDgram + m_para.maxTrSize;
            auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
            memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

            if (service == TransitionId::Enable) {
                // Drain the encQueue and repopulate the freelist
                m_det.udpReceiver()->drain();

                // Reset the interpolator
                m_interpolator.reset();

                m_running = true;
            }
            else if (service == TransitionId::Disable) {
                m_running = false;
            }
        }
    }
    _sendToTeb(*pebbleDg, pebbleIdx);

    uint32_t evtIdx;
    auto rc = m_evtQueue.try_pop(evtIdx);         // Actually consume the pebble index
    if (rc)  assert(evtIdx == pebbleIdx);
}

void UdpDrp::_handleL1Accept(EbDgram& pgpDg, const encoder_frame_t& frame,
                                 uint32_t *rawValue, uint32_t *interpolatedValue)
{
    uint32_t evtIdx;
    m_evtQueue.try_pop(evtIdx);         // Actually consume the element

    auto bufEnd = (char*)&pgpDg + pool.pebble.bufferSize();

    m_det.event_(pgpDg, bufEnd, frame, rawValue, interpolatedValue);

    if (!m_para.interpolating || (pgpDg.readoutGroups() & (1 << m_para.slowGroup))) {
        m_det.udpReceiver()->consume();
    }

    ++m_nMatch;

    _sendToTeb(pgpDg, evtIdx);
}

void UdpDrp::_timeout(ms_t timeout)
{
    // Try to clear out as many of the older queue entries as we can in one go
    while (true) {
        // Time out older pending PGP datagrams
        uint32_t index;
        if (!m_evtQueue.peek(index))  break;

        EbDgram& dgram = *reinterpret_cast<EbDgram*>(pool.pebble[index]);
        if (m_pgp.age(dgram.time) < timeout)  break;

        logging::debug("Event timed out!! "
                       "TimeStamp:  %u.%09u [0x%08x%04x.%05x], age %ld ms, svc %u",
                       dgram.time.seconds(), dgram.time.nanoseconds(),
                       dgram.time.seconds(), (dgram.time.nanoseconds()>>16)&0xfffe, dgram.time.nanoseconds()&0x1ffff,
                       Eb::latency<ms_t>(dgram.time), dgram.service());

        if (dgram.service() == TransitionId::L1Accept) {
            // No encoder data so mark event as damaged
            dgram.xtc.damage.increase(Damage::TimedOut);
            ++m_nTimedOut;

            uint32_t idx;
            auto rc = m_evtQueue.try_pop(idx);              // Actually consume the element
            if (rc)  assert(idx == index);

            _sendToTeb(dgram, index);
        }
        else {
            _handleTransition(index, &dgram);
        }
    }
}

void UdpDrp::_sendToTeb(const EbDgram& dgram, uint32_t index)
{
    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = (dgram.service() == TransitionId::L1Accept)
                         ? pool.pebble.bufferSize()
                         : m_para.maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", TransitionId::name(dgram.service()), size, maxSize);
        throw "Dgram overflowed buffer";
    }

    auto l3InpBuf = tebContributor().fetch(index);
    EbDgram* l3InpDg = new(l3InpBuf) EbDgram(dgram);
    if (l3InpDg->isEvent()) {
        auto trgPrimitive = triggerPrimitive();
        if (trgPrimitive) { // else this DRP doesn't provide input
            const void* bufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + trgPrimitive->size();
            trgPrimitive->event(pool, index, dgram.xtc, l3InpDg->xtc, bufEnd); // Produce
        }
    }
    tebContributor().process(l3InpDg);
}


UdpApp::UdpApp(UdpParameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_para(para),
    m_pool(para),
    m_unconfigure(false)
{
    Py_Initialize();  // Must be called before creating the UdpEncoder

    m_det = std::make_unique<UdpEncoder>(m_para, m_pool);
    m_drp = std::make_unique<UdpDrp>(para, m_pool, *m_det, context());

    logging::info("Ready for transitions");
}

UdpApp::~UdpApp()
{
    // Try to take things down gracefully when an exception takes us off the normal path
    handleReset(json({}));

    Py_Finalize();
}

void UdpApp::_disconnect()
{
    m_drp->disconnect();
    m_det->disconnect();
}

void UdpApp::_unconfigure()
{
    m_drp->pool.shutdown();  // Release Tr buffer pool
    m_drp->unconfigure();
    m_det->unconfigure();
    m_unconfigure = false;
}

json UdpApp::connectionInfo(const json& msg)
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

void UdpApp::connectionShutdown()
{
    static_cast<Detector&>(*m_det).connectionShutdown();
    m_drp->shutdown();
}

void UdpApp::_error(const std::string& which, const json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void UdpApp::handleConnect(const json& msg)
{
    std::string errorMsg = m_drp->connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error(("DrpBase::connect: " + errorMsg).c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    m_det->nodeId = m_drp->nodeId();
    unsigned rc = m_det->connect(msg, std::to_string(getId()), errorMsg);
    if (!errorMsg.empty()) {
        if (!rc) {
            logging::warning(("UdpDetector::connect: " + errorMsg).c_str());
            json warning = createAsyncWarnMsg(m_para.alias, errorMsg);
            reply(warning);
        }
        else {
            logging::error(("UdpDetector::connect: " + errorMsg).c_str());
            _error("connect", msg, errorMsg);
            return;
        }
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
            std::string errorMsg = "Failed transition phase 1";
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
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else {
            m_drp->runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
        }
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp->endrun(phase1Info);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
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
        m_det->reset(); // needed?
        logging::debug("handlePhase1 enable complete");
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void UdpApp::handleReset(const json& msg)
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
    logging::error("Failed to set socket receive buffer to %u bytes: %m", UDP_RCVBUF_SIZE);
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
    Drp::UdpParameters para;
    std::string kwargs_str;
    int c;
    while((c = getopt(argc, argv, "p:L:o:l:D:S:C:d:u:k:P:M:v")) != EOF) {
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
                para.detType = optarg;  // Defaults to 'encoder'
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
                           : kwargs_str + "," + optarg;
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'M':
                para.prometheusDir = optarg;
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
            if (kwargs.first == "match_tmo_ms")   continue;
            if (kwargs.first == "slowGroup")      continue;
            if (kwargs.first == "encTprAlias")    continue;
            logging::critical("Unrecognized kwarg '%s=%s'\n",
                              kwargs.first.c_str(), kwargs.second.c_str());
            return 1;
        }
        Drp::UdpApp app(para);
        app.run();
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
}

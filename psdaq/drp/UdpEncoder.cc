#include "UdpEncoder.hh"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <eigen3/Eigen/QR>

#ifdef NDEBUG
#undef NDEBUG
#endif

#define MAX_ENC_VALUES 2

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
#include <sys/param.h>
#include <sys/select.h>
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
#include "psdaq/service/fast_monotonic_clock.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

static const XtcData::TimeStamp TimeMax(std::numeric_limits<unsigned>::max(),
                                        std::numeric_limits<unsigned>::max());

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
      hardwareID,
      innerCount
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
       NameVec.push_back({"innerCount", XtcData::Name::UINT16});
   }
} RawDef;

namespace Drp {

UdpReceiver::UdpReceiver(const Parameters&           para,
                         SPSCQueue<XtcData::Dgram*>& pvQueue,
                         SPSCQueue<XtcData::Dgram*>& bufferFreelist,
                         const bool& firstReadout):
    m_para          (para),
    m_pvQueue       (pvQueue),
    m_bufferFreelist(bufferFreelist),
    m_firstReadout(firstReadout),
    m_enc_values    (MAX_ENC_VALUES),
    m_enc_times     (MAX_ENC_VALUES),
    m_num_enc_values(0),
    m_terminate     (false),
    m_outOfOrder    (false),
    m_missingData   (false),
    m_notifySocket  {&m_context, ZMQ_PUSH},
    m_nUpdates      (0),
    m_encoderValT1  (0),
    m_encoderValT2  (0),
    m_nMissed       (0)
{
    // ZMQ socket for reporting errors
    m_notifySocket.connect({"tcp://" + m_para.collectionHost + ":" + std::to_string(CollectionApp::zmq_base_port + m_para.partition)});

    // UDP socket for receiving data from DATA PORT after interpolation
    _dataFd = createUdpSocket(UdpEncoder::DefaultDataPort);
    logging::debug("createUdpSocket(%d) returned %d (data fd)", UdpEncoder::DefaultDataPort, _dataFd);

    // UDP socket for receiving data from INTERPOLATE PORT before interpolation
    _interpolateFd = createUdpSocket(UdpEncoder::DefaultInterpolatePort);
    logging::debug("createUdpSocket(%d) returned %d (interpolate fd)", UdpEncoder::DefaultInterpolatePort, _interpolateFd);
}

UdpReceiver::~UdpReceiver()
{
    if (_dataFd > 0) {
        close(_dataFd);
    }
    if (_interpolateFd > 0) {
        close(_interpolateFd);
    }
}

void UdpReceiver::start()
{
    m_resetHwCount = true;

    m_terminate.store(false, std::memory_order_release);

    m_udpReceiverThread = std::thread{&UdpReceiver::_udpReceiver, this};

    _loopbackInit();

    logging::info("%s started", name().c_str());
}

void UdpReceiver::stop()
{
    m_terminate.store(true, std::memory_order_release);

    if (m_udpReceiverThread.joinable()) {
        m_udpReceiverThread.join();
    }

    logging::info("%s stopped", name().c_str());
}

void UdpReceiver::_loopbackInit()
{
    logging::debug("%s (port = %d)", __PRETTY_FUNCTION__, m_para.loopbackPort);

    if (true) {
        m_loopbackFd = socket(AF_INET,SOCK_DGRAM, 0);
        if (m_loopbackFd == -1) {
            perror("socket");
            logging::error("failed to create loopback socket");
        }

        // bind the socket to any valid IP address and a specific port */
        bzero(&m_loopbackAddr, sizeof(m_loopbackAddr));
        m_loopbackAddr.sin_family = AF_INET;
        m_loopbackAddr.sin_addr.s_addr = htonl(INADDR_ANY);
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

#if 0
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
        logging::error("%s: failed to send to loopback socket %d", __PRETTY_FUNCTION__, m_loopbackFd);
    } else {
        logging::debug("%s: sent = %d", __PRETTY_FUNCTION__, sent);
    }
}
#endif

void UdpReceiver::_udpReceiver()
{
    logging::info("%s: thread started", __PRETTY_FUNCTION__);

    fd_set readfds, masterfds;
    struct timeval timeout;

    FD_ZERO(&masterfds);
    FD_SET(_dataFd, &masterfds);
    FD_SET(_interpolateFd, &masterfds);
    int nfds = MAX(_dataFd, _interpolateFd) + 1;
    logging::debug("%s: nfds = %d", __PRETTY_FUNCTION__, nfds);

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
        if (select(nfds, &readfds, NULL, NULL, &timeout) < 0) {
            logging::error("select: error");
            break;
        }
        if (m_terminate.load(std::memory_order_relaxed)) {
            // shutting down
            break;
        }
        logging::debug("%s check read FDs", __PRETTY_FUNCTION__);
        if (FD_ISSET(_interpolateFd, &readfds)) {
            logging::debug("%s interpolate read FD is set", __PRETTY_FUNCTION__);
            interpolate();
        }
        if (FD_ISSET(_dataFd, &readfds)) {
            logging::debug("%s data read FD is set", __PRETTY_FUNCTION__);
            process();
        }
    }
    logging::info("%s: thread finished", __PRETTY_FUNCTION__);
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

void UdpReceiver::_read(XtcData::Dgram& dgram)
{
    const void* bufEnd = dgram.xtc.payload() + sizeof(encoder_frame_t);
    encoder_frame_t* frame = (encoder_frame_t*)dgram.xtc.alloc(sizeof(encoder_frame_t), bufEnd);
    bool missing = false;

    // read from the udp socket that triggered select()
    int rv = _readFrame(_dataFd, frame, missing);

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
                     "Missing data for frame %hu", frame->header.frameCount);
            setMissingData(errmsg);
        }
    }

    logging::debug("%s: frame=%hu  inner=%hu  encoderValue=%u  timing=%u  scale=%u  scaleDenom=%u  mode=%u  error=%u  version=%u.%u.%u",
                   __PRETTY_FUNCTION__,
#if 0
                   ntohs(frame->header.frameCount),
                   ntohs(frame->header.innerCount),
                   ntohl(frame->channel[0].encoderValue),
                   ntohl(frame->channel[0].timing),
                   (unsigned) ntohs(frame->channel[0].scale),
                   (unsigned) ntohs(frame->channel[0].scaleDenom),
                   (unsigned) frame->channel[0].mode,
                   (unsigned) frame->channel[0].error,
                   (unsigned) ntohs(frame->header.majorVersion),
#else
                   frame->header.frameCount,
                   frame->header.innerCount,
                   frame->channel[0].encoderValue,
                   frame->channel[0].timing,
                   (unsigned) frame->channel[0].scale,
                   (unsigned) frame->channel[0].scaleDenom,
                   (unsigned) frame->channel[0].mode,
                   (unsigned) frame->channel[0].error,
                   (unsigned) frame->header.majorVersion,
#endif
                   (unsigned) frame->header.minorVersion,
                   (unsigned) frame->header.microVersion);

    // reset frame counter
    if (m_resetHwCount) {
        m_count = 0;
        m_countOffset = frame->header.frameCount - 1;
        m_resetHwCount = false;
    }

    if (frame->header.innerCount == 0) {
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
    }

    // record damage
    if (m_outOfOrder) {
        dgram.xtc.damage.increase(XtcData::Damage::OutOfOrder);
    }
}

void UdpReceiver::process()
{
    ++m_nUpdates;
    logging::debug("%s process", name().c_str());
//  logging::debug("%s: m_preInterpolation = %s", __PRETTY_FUNCTION__, m_preInterpolation ? "true" : "false");

    XtcData::Dgram* dgram;
    if (m_bufferFreelist.try_pop(dgram)) { // If a buffer is available...

        dgram->xtc = {{XtcData::TypeId::Parent, 0}, {0}};

        _read(*dgram);                     // read the frame into the Dgram

        m_pvQueue.push(dgram);
    }
    else {
        logging::error("%s: buffer not available, frame dropped", __PRETTY_FUNCTION__);
        ++m_nMissed;                       // Else count it as missed
        (void) _junkFrame(_dataFd);
    }
}

void UdpReceiver::interpolate() {
    logging::debug("%s: m_loopbackFd = %d, _interpolateFd = %d", __PRETTY_FUNCTION__, m_loopbackFd, _interpolateFd);
    static encoder_frame_t newFrame, oldFrame;
    std::vector<double> coeff;
    bool coeff_ready = false;

    if (! m_firstReadout) {
        oldFrame = newFrame;
    }
    // read from the udp socket that triggered select()
    ssize_t recvlen = recvfrom(_interpolateFd, &newFrame, sizeof(encoder_frame_t), MSG_DONTWAIT, 0, 0);
    // check length
    if (recvlen != (ssize_t) sizeof(encoder_frame_t)) {
        // length is wrong
        if (recvlen == -1) {
            perror("recvfrom");
        }
        logging::error("received UDP length %zd, expected %zd", recvlen, sizeof(encoder_frame_t));
    } else {
        // length is OK
        logging::debug("%s: recvfrom() length %zd is OK", __PRETTY_FUNCTION__, recvlen);

        if (m_firstReadout) {
            m_encoderValT2 = ntohl(newFrame.channel[0].encoderValue);
            logging::debug("%s: first sample: m_encoderValT2 = %7u", __PRETTY_FUNCTION__, m_encoderValT2);
        } else {

// ---------------------------------------- FIT ---------------------------------------------------------

            m_enc_values[m_num_enc_values%MAX_ENC_VALUES] = (double)ntohl(newFrame.channel[0].encoderValue);

            for (unsigned uu = 0; uu <= Order; uu++) {
                m_enc_times[(m_num_enc_values - uu) % MAX_ENC_VALUES] = (Order - uu) * TriggerRatio;
            }

            m_num_enc_values++;

            logging::debug("***m_enc_times:  %6.3f %6.3f", m_enc_times[0], m_enc_times[1]);
            logging::debug("***m_enc_values: %6.3f %6.3f", m_enc_values[0], m_enc_values[1]);
            logging::debug("***event %d", m_num_enc_values);
            if (m_num_enc_values > Order) {
                logging::debug("***fit");
                _polyfit(m_enc_times, m_enc_values, coeff, Order);
                logging::debug("***coeff:        %6.3f %6.3f", coeff[0], coeff[1]);
                coeff_ready = true;
            }
        }

// ------------------------------------------------------------------------------------------------------

        if (coeff_ready) {
            // interpolate
            for (uint16_t ii = 1; ii < TriggerRatio; ii++) {
                // set innercount
                oldFrame.header.innerCount = htons(ii);

                // set interpolated encoder value
                double q = (double) ii / (TriggerRatio * Order);
                double vfitted = coeff[0];
                for (unsigned ord = 1; ord <= Order; ord++) {
                    vfitted += (coeff[ord] * pow(q, ord) * TriggerRatio);
                }
                logging::debug("***vfitted: %6.3f  (q=%6.3f)", vfitted, q);
                oldFrame.channel[0].encoderValue = htonl(round(vfitted));

                // copy hardware ID
                memcpy(oldFrame.header.hardwareID, newFrame.header.hardwareID, 15);
                oldFrame.header.hardwareID[15] = '\0';

                // copy version
                oldFrame.header.majorVersion = newFrame.header.majorVersion;
                oldFrame.header.minorVersion = newFrame.header.minorVersion;
                oldFrame.header.microVersion = newFrame.header.microVersion;

                // send interpolated frame
                int sent = sendto(m_loopbackFd, (void *)&oldFrame, sizeof(oldFrame), 0,
                              (struct sockaddr *)&m_loopbackAddr, sizeof(m_loopbackAddr));

                if (sent == -1) {
                    perror("sendto");
                    logging::error("%s: failed to send to loopback socket %d", __PRETTY_FUNCTION__, m_loopbackFd);
                } else {
                    if (sent != sizeof(encoder_frame_t)) {
                        logging::error("%s: sendto() returned %d", __PRETTY_FUNCTION__, sent);
                    }
                    logging::debug("%s: %2c frameCount=%-5u innerCount=%-5u encoderValue=%-5u", __PRETTY_FUNCTION__,
                                   (ntohs(oldFrame.header.innerCount) == 0) ? '*' : ' ',
                                   ntohs(oldFrame.header.frameCount),
                                   ntohs(oldFrame.header.innerCount),
                                   ntohl(oldFrame.channel[0].encoderValue));
                }
            }
        }

        // send the original UDP frame (with innerCount=0)
        newFrame.header.innerCount = htons(0);
        int sent = sendto(m_loopbackFd, (void *)&newFrame, sizeof(newFrame), 0,
                      (struct sockaddr *)&m_loopbackAddr, sizeof(m_loopbackAddr));

        if (sent == -1) {
            perror("sendto");
            logging::error("%s: failed to send to loopback socket %d", __PRETTY_FUNCTION__, m_loopbackFd);
        } else {
            logging::debug("%s: %2c frameCount=%-5u innerCount=%-5u encoderValue=%-5u sendto()=%d", __PRETTY_FUNCTION__,
                           (ntohs(newFrame.header.innerCount) == 0) ? '*' : ' ',
                           ntohs(newFrame.header.frameCount),
                           ntohs(newFrame.header.innerCount),
                           ntohl(newFrame.channel[0].encoderValue), sent);
        }
    }
}

int UdpReceiver::_readFrame(int fd, encoder_frame_t *frame, bool& missing)
{
    logging::debug("entered %s", __PRETTY_FUNCTION__);
    int rv = 0;
    ssize_t recvlen;

    // peek data
    recvlen = recvfrom(fd, frame, sizeof(encoder_frame_t), MSG_DONTWAIT | MSG_PEEK, 0, 0);
    // check length
    if (recvlen != (ssize_t) sizeof(encoder_frame_t)) {
        if (recvlen == -1) {
            perror("recvfrom(MSG_PEEK)");
        }
        logging::error("received UDP length %zd, expected %zd", recvlen, sizeof(encoder_frame_t));
        // TODO discard frame of the wrong size
    } else {
        // length is OK
        logging::debug("%s: recvfrom() length %zd is OK", __PRETTY_FUNCTION__, recvlen);

        // byte swap
        frame->header.frameCount = ntohs(frame->header.frameCount);
        frame->header.innerCount = ntohs(frame->header.innerCount);
        frame->header.majorVersion = ntohs(frame->header.majorVersion);
    }
    if ((frame->header.innerCount == 0) && (m_resetHwCount == false)) {
        uint16_t expect16 = (uint16_t)(1 + m_count + m_countOffset);
        if (frame->header.frameCount != expect16) {
            // frame count doesn't match
            logging::error("recvfrom(MSG_PEEK) frameCount %hu (expected %hu)\n", frame->header.frameCount, expect16);
            // trigger MissingData damage
            missing = true;
            // return empty frame with expected frame count
            bzero(frame, sizeof(encoder_frame_t));
            frame->header.frameCount = expect16;
            frame->header.majorVersion = htons(UdpEncoder::MajorVersion);
            frame->header.minorVersion = UdpEncoder::MinorVersion;
            frame->header.microVersion = UdpEncoder::MicroVersion;
            return (0);
        }
    }

    // read data
    recvlen = recvfrom(fd, frame, sizeof(encoder_frame_t), MSG_DONTWAIT, 0, 0);
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
        frame->header.innerCount = ntohs(frame->header.innerCount);
        frame->header.majorVersion = ntohs(frame->header.majorVersion);
        if (frame->header.majorVersion != UdpEncoder::MajorVersion) {
            logging::error("%s: found majorVersion %u, expected %d", __PRETTY_FUNCTION__,
            frame->header.majorVersion, UdpEncoder::MajorVersion);
        }
        frame->channel[0].encoderValue = ntohl(frame->channel[0].encoderValue);
        frame->channel[0].timing = ntohl(frame->channel[0].timing);
        frame->channel[0].scale = ntohs(frame->channel[0].scale);
        frame->channel[0].scaleDenom = ntohs(frame->channel[0].scaleDenom);

        logging::debug("     frameCount    %-7u", frame->header.frameCount);
        logging::debug("     innerCount    %-7u", frame->header.innerCount);
        logging::debug("     version       %u.%u.%u", frame->header.majorVersion,
                                                      frame->header.minorVersion,
                                                      frame->header.microVersion);
        char buf[16];
        snprintf(buf, sizeof(buf), "%s", frame->header.hardwareID);
        buf[15] = 0;
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

int UdpReceiver::_junkFrame(int fd)
{
    int rv = 0;
    ssize_t recvlen;
    encoder_frame_t junk;

    // read data
    recvlen = recvfrom(fd, &junk, sizeof(junk), MSG_DONTWAIT, 0, 0);
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

int UdpReceiver::drainFd(int fd)
{
  int rv = 0;
  unsigned count = 0;
  encoder_frame_t junk;

  if (fd > 0) {
    while ((rv = recvfrom(fd, &junk, sizeof(junk), MSG_DONTWAIT, 0, 0)) > 0) {
      if (rv == -1) {
        perror("recvfrom");
        break;
      }
      ++ count;
    }
    if (count > 0) {
      logging::debug("%s: drained %u frames\n", __PRETTY_FUNCTION__, count);
    }
  }

  return (rv);
}

int UdpReceiver::reset()
{
  int rv1 = -1;
  int rv2 = -1;

  if (_dataFd > 0) {
    // drain input buffers
    rv1 = drainFd(_dataFd);
  }
  if (_interpolateFd > 0) {
    // drain interpolation buffers
    rv2 = drainFd(_interpolateFd);
  }
  return ((rv1 == 0) && (rv2 == 0)) ? 0 : -1;
}

class Pgp : public PgpReader
{
public:
    Pgp(const Parameters& para, DrpBase& drp, Detector* det, const bool& running, const bool& preInterpolation) :
        PgpReader(para, drp.pool, MAX_RET_CNT_C, 32),
        m_det(det), m_tebContributor(drp.tebContributor()), m_running(running), m_preInterpolation(preInterpolation),
        m_available(0), m_current(0), m_nDmaRet(0)
    {
        m_nodeId = drp.nodeId();
        if (drp.pool.setMaskBytes(para.laneMask, 0)) {
            logging::error("Failed to allocate lane/vc");
        }
    }

    Pds::EbDgram* next(uint32_t& evtIndex);
    const uint64_t nDmaRet() { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    Detector* m_det;
    Pds::Eb::TebContributor& m_tebContributor;
    static const int MAX_RET_CNT_C = 100;
    const bool& m_running;
    const bool& m_preInterpolation;
    int32_t m_available;
    int32_t m_current;
    unsigned m_nodeId;
    uint64_t m_nDmaRet;
};

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

Pds::EbDgram* Pgp::next(uint32_t& evtIndex)
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

    Pds::EbDgram* dgram = _handle(evtIndex);
    m_current++;
    return dgram;
}


UdpEncoder::UdpEncoder(Parameters& para, DrpBase& drp) :
    XpmDetector     (&para, &drp.pool),
    m_drp           (drp),
    m_evtQueue      (drp.pool.nbuffers()),
    m_pvQueue       (128),                // Formerly 8
    m_bufferFreelist(m_pvQueue.size()),
    m_terminate     (false),
    m_running       (false),
    m_preInterpolation (true)
{
}

unsigned UdpEncoder::connect(std::string& msg)
{
    try {
        m_udpReceiver = std::make_shared<UdpReceiver>(*m_para, m_pvQueue, m_bufferFreelist, m_firstReadout);
    }
    catch(std::string& error) {
        logging::error("Failed to create UdpReceiver: %s", error.c_str());
        m_udpReceiver.reset();
        msg = error;
        return 1;
    }

    return 0;
}

unsigned UdpEncoder::disconnect()
{
    m_udpReceiver.reset();
    return 0;
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

    addNames(0, xtc, bufEnd);

    // (Re)initialize the queues
    m_pvQueue.startup();
    m_evtQueue.startup();
    m_bufferFreelist.startup();
    size_t bufSize = sizeof(XtcData::Dgram) + sizeof(encoder_frame_t);
    m_buffer.resize(m_pvQueue.size() * bufSize);
    for(unsigned i = 0; i < m_pvQueue.size(); ++i) {
        m_bufferFreelist.push(reinterpret_cast<XtcData::Dgram*>(&m_buffer[i * bufSize]));
    }

    m_terminate.store(false, std::memory_order_release);

    m_workerThread = std::thread{&UdpEncoder::_worker, this};

    return 0;
}

unsigned UdpEncoder::unconfigure()
{
    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    m_pvQueue.shutdown();
    m_evtQueue.shutdown();
    m_bufferFreelist.shutdown();
    m_namesLookup.clear();   // erase all elements

    return 0;
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
    m_exporter->add("drp_update_rate", labels, Pds::MetricType::Rate,
                    [&](){return m_udpReceiver ? m_udpReceiver->nUpdates() : 0;});
    m_nMatch = 0;
    m_exporter->add("drp_match_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nMatch;});
    m_exporter->add("drp_miss_count", labels, Pds::MetricType::Counter,
                    [&](){return m_udpReceiver ? m_udpReceiver->nMissed() : 0;});
    m_nTimedOut = 0;
    m_exporter->add("drp_timeout_count", labels, Pds::MetricType::Counter,
                    [&](){return m_nTimedOut;});

    m_exporter->add("drp_worker_input_queue", labels, Pds::MetricType::Gauge,
                    [&](){return m_evtQueue.guess_size();});
    m_exporter->constant("drp_worker_queue_depth", labels, m_evtQueue.size());

    // Borrow this for awhile
    m_exporter->add("drp_worker_output_queue", labels, Pds::MetricType::Gauge,
                    [&](){return m_pvQueue.guess_size();});

    Pgp pgp(*m_para, m_drp, this, m_running, m_preInterpolation);

    m_exporter->add("drp_num_dma_ret", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nDmaRet();});
    m_exporter->add("drp_pgp_byte_rate", labels, Pds::MetricType::Rate,
                    [&](){return pgp.dmaBytes();});
    m_exporter->add("drp_dma_size", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.dmaSize();});
    m_exporter->add("drp_th_latency", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.latency();});
    m_exporter->add("drp_num_dma_errors", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nDmaErrors();});
    m_exporter->add("drp_num_no_common_rog", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nNoComRoG();});
    m_exporter->add("drp_num_th_error", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nTmgHdrError();});
    m_exporter->add("drp_num_pgp_jump", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nPgpJumps();});
    m_exporter->add("drp_num_no_tr_dgram", labels, Pds::MetricType::Gauge,
                    [&](){return pgp.nNoTrDgrams();});

    const uint64_t msTmo = m_para->kwargs.find("match_tmo_ms") != m_para->kwargs.end()
                         ? std::stoul(m_para->kwargs["match_tmo_ms"])
                         : 100;

    enum TmoState { None, Started, Finished };
    TmoState tmoState(TmoState::None);
    const std::chrono::microseconds tmo(int(1.1 * m_drp.tebPrms().maxEntries * 14/13));
    auto tInitial = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC);

    m_udpReceiver->start();

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        Pds::EbDgram* dgram = pgp.next(index);
        if (dgram) {
            tmoState = TmoState::None;
            m_nEvents++;
            logging::debug("Worker thread: m_nEvents = %d", m_nEvents);

            XtcData::TransitionId::Value service = dgram->service();
#if 0
            if (service == XtcData::TransitionId::L1Accept) {
                if (m_para->loopbackPort) {
                    m_udpReceiver->loopbackSend();        // LOOPBACK TEST
                }
            }
#endif
            // Also queue SlowUpdates to keep things in time order
            if ((service == XtcData::TransitionId::L1Accept) ||
                (service == XtcData::TransitionId::SlowUpdate)) {
                m_evtQueue.push(index);

                //printf("                         PGP: %u.%09u\n",
                //       dgram->time.seconds(), dgram->time.nanoseconds());

                _matchUp();

                // Prevent PGP events from stacking up by by timing them out.
                // The maximum timeout is < the TEB event build timeout to keep
                // prompt contributions from timing out before latent ones arrive.
                // If the PV is updating, _timeout() never finds anything to do.
                XtcData::TimeStamp timestamp;
                const uint64_t nsTmo = msTmo * 1000000;
                _timeout(timestamp.from_ns(dgram->time.to_ns() - nsTmo));
            }
            else {
                // Find the transition dgram in the pool and initialize its header
                Pds::EbDgram* trDgram = m_pool->transitionDgrams[index];
                const void*   bufEnd  = (char*)trDgram + m_para->maxTrSize;
                if (!trDgram)  continue; // Can happen during shutdown
                *trDgram = *dgram;
                // copy the temporary xtc created on phase 1 of the transition
                // into the real location
                XtcData::Xtc& trXtc = transitionXtc();
                trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                if (service == XtcData::TransitionId::Enable) {
                    m_running = true;
                    m_preInterpolation = m_firstReadout = true;
                }
                else if (service == XtcData::TransitionId::Disable) { // Sweep out L1As
                    m_running = false;
                    logging::debug("Sweeping out L1Accepts and SlowUpdates");
                    _timeout(TimeMax);
                }

                _sendToTeb(*dgram, index);
            }
        }
        else {
            if (tmoState == TmoState::None) {
                tmoState = TmoState::Started;
                tInitial = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC);
            } else {
                if (Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC) - tInitial > tmo) {
                    if (tmoState != TmoState::Finished) {
                        m_drp.tebContributor().timeout();
                        tmoState = TmoState::Finished;
                    }
                }
            }
        }
    }

    m_udpReceiver->stop();

    // Flush the DMA buffers
    pgp.flush();

    logging::info("Worker thread finished");
}

void UdpEncoder::_matchUp()
{
    while (true) {
        XtcData::Dgram* pvDg;
        if (!m_pvQueue.peek(pvDg))  break;

        uint32_t evtIdx;
        if (!m_evtQueue.peek(evtIdx))  break;

        Pds::EbDgram* pgpDg = reinterpret_cast<Pds::EbDgram*>(m_pool->pebble[evtIdx]);

        _handleMatch  (*pvDg, *pgpDg);
    }
}

void UdpReceiver::_polyfit(const std::vector<double> &t,
                          const std::vector<double> &v,
                          std::vector<double> &coeff,
                          unsigned order)
{
    // Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
    Eigen::MatrixXd T(t.size(), order + 1);
    Eigen::VectorXd V = Eigen::VectorXd::Map(&v.front(), v.size());
    Eigen::VectorXd result;

    // check to make sure inputs are correct
    assert(t.size() == v.size());
    assert(t.size() >= order + 1);

    // Populate the matrix
    for (size_t i = 0 ; i < t.size(); ++i) {
        for (size_t j = 0; j < order + 1; ++j) {
            T(i, j) = pow(t.at(i), j);
        }
    }
    //std::cout<<T<<std::endl;

    // Solve for linear least square fit
    result  = T.householderQr().solve(V);
    coeff.resize(order+1);
    for (unsigned k = 0; k < order+1; k++) {
        coeff[k] = result[k];
    }
}

void UdpEncoder::_event(XtcData::Dgram& dgram, const void* const bufEnd, encoder_frame_t& frame)
{
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

    // ...innerCount
    raw.set_value(RawDef::innerCount, frame.header.innerCount);
}

void UdpEncoder::_handleMatch(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg)
{
    logging::debug("%s: pgpDg rogs = %04hx", __PRETTY_FUNCTION__,
                   pgpDg.readoutGroups());

    if (m_preInterpolation && (pgpDg.readoutGroups() & (1 << TriggerReadoutGroup))) {
        logging::info("%s: rogs p5 first MATCH!", __PRETTY_FUNCTION__);
        m_preInterpolation = false;
    }

    uint32_t evtIdx;
    m_evtQueue.try_pop(evtIdx);         // Actually consume the element

    XtcData::Dgram* dgram;
    if (pgpDg.service() == XtcData::TransitionId::L1Accept) {
        pgpDg.xtc.damage.increase(pvDg.xtc.damage.value());
        auto bufEnd  = (char*)&pgpDg + m_pool->pebble.bufferSize();
        if (m_preInterpolation) {
            pgpDg.xtc.damage.increase(XtcData::Damage::MissingData);
            logging::info("%s: pre-interpolation MissingData damage", __PRETTY_FUNCTION__);
        }
        _event(pgpDg, bufEnd, *(encoder_frame_t*)(pvDg.xtc.payload()));

        m_pvQueue.try_pop(dgram);       // Actually consume the element
        m_bufferFreelist.push(dgram);   // Return buffer to freelist

        ++m_nMatch;
    }

    _sendToTeb(pgpDg, evtIdx);
}

void UdpEncoder::_timeout(const XtcData::TimeStamp& timestamp)
{
    while (true) {
        uint32_t index;
        if (!m_evtQueue.peek(index)) {
            break;
        }

        Pds::EbDgram& dgram = *reinterpret_cast<Pds::EbDgram*>(m_pool->pebble[index]);
        if (dgram.time > timestamp) {
            break;                  // dgram is newer than the timeout timestamp
        }

        uint32_t idx;
        m_evtQueue.try_pop(idx);        // Actually consume the element
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
            Pds::EbDgram* trDg = m_pool->transitionDgrams[index];
            *trDg = dgram;              // Initialized Xtc, possibly w/ damage
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

UdpApp::UdpApp(Parameters& para) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_udpDetector(std::make_unique<UdpEncoder>(m_para, m_drp)),
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
    m_udpDetector->disconnect();
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
    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect");
        logging::error("%s", errorMsg.c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    m_det->nodeId = msg["body"]["drp"][std::to_string(getId())]["drp_id"];
    m_det->connect(msg, std::to_string(getId()));

    unsigned rc = m_udpDetector->connect(errorMsg);
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
        else {
            m_drp.runInfoData(xtc, bufEnd, m_det->namesLookup(), runInfo);
        }
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
    para.loopbackPort = 5006;   // UdpEncoder::DefaultDataPort

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
            if (kwargs.first == "match_tmo_ms")   continue;
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

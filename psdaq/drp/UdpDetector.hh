#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <assert.h>
#include <cstdint>
#include "DrpBase.hh"
#include "XpmDetector.hh"
#include "spscqueue.hh"
#include "psdaq/service/Collection.hh"

#define UDP_RCVBUF_SIZE 10000

namespace Drp {

// encoder header: 32 bytes
typedef struct {
    uint16_t    frameCount;         // network byte order
    uint8_t     reserved1[2];
    uint16_t    majorVersion;       // network byte order
    uint8_t     minorVersion;
    uint8_t     microVersion;
    char        hardwareID[16];
    uint8_t     reserved2;
    uint8_t     channelMask;
    uint8_t     errorMask;
    uint8_t     mode;
    uint8_t     reserved3[4];
} encoder_header_t;

static_assert(sizeof(encoder_header_t) == 32, "Data structure encoder_header_t is not size 32");


// encoder channel: 32 bytes
typedef struct {
    uint32_t    encoderValue;       // network byte order
    uint32_t    timing;             // network byte order
    uint16_t    scale;              // network byte order
    char        hardwareID[16];
    uint8_t     reserved1;
    uint8_t     channel;
    uint8_t     error;
    uint8_t     mode;
    uint8_t     reserved2[2];
} encoder_channel_t;

static_assert(sizeof(encoder_channel_t) == 32, "Data structure encoder_channel_t is not size 32");


// encoder frame: 64 bytes
typedef struct {
    encoder_header_t    header;
    encoder_channel_t   channel[1];
} encoder_frame_t;

static_assert(sizeof(encoder_frame_t) == 64, "Data structure encoder_frame_t is not size 64");
class UdpDetector;

class UdpMonitor
{
public:
    UdpMonitor(Parameters& para) :
      m_para                  (para),
      m_udpDetector           (nullptr)
    {
    }
public:
    void onConnect();
    void onDisconnect();
    const std::string name() const { return "encoder"; }    // FIXME
    bool connected() const { return _connected; }
public:
    bool ready(UdpDetector* udpDetector);
    void clear() { m_udpDetector = nullptr; }
protected:
    bool                _connected;
private:
    Parameters&         m_para;
    UdpDetector*        m_udpDetector;
};


class UdpDetector : public XpmDetector
{
public:
    UdpDetector(Parameters& para, std::shared_ptr<UdpMonitor>& udpMonitor, DrpBase& drp);
    ~UdpDetector();
  //    std::string sconfigure(const std::string& config_alias, XtcData::Xtc& xtc);
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
    unsigned unconfigure();
    void setOutOfOrder(std::string errMsg);
    bool getOutOfOrder() { return (m_outOfOrder); }
    void process();
    void addNames(unsigned segment, XtcData::Xtc& xtc);
    int drainFd(int fd);
    int reset();
    enum { DefaultDataPort = 5006 };
private:
    int _readFrame(encoder_frame_t *frame);
    void _loopbackInit();
    void _loopbackFini();
    void _loopbackSend();
    void _worker();
    void _udpReceiver();
    void _timeout(const XtcData::TimeStamp& timestamp);
    void _matchUp();
    void _handleMatch(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg);
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    enum {RawNamesIndex = NamesIndex::BASE, InfoNamesIndex};
    enum { DiscardBufSize = 10000 };
    int m_loopbackFd;
    struct sockaddr_in m_loopbackAddr;
    uint16_t m_loopbackFrameCount;
    DrpBase& m_drp;
    std::shared_ptr<UdpMonitor> m_udpMonitor;
    std::thread m_workerThread;
    std::thread m_udpReceiverThread;
    SPSCQueue<uint32_t> m_pgpQueue;
    SPSCQueue<XtcData::Dgram*> m_pvQueue;
    SPSCQueue<XtcData::Dgram*> m_bufferFreelist;
    std::vector<uint8_t> m_buffer;
    std::atomic<bool> m_terminate;
    std::atomic<bool> m_running;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    uint64_t m_nEvents;
    uint64_t m_nUpdates;
    uint64_t m_nMissed;
    uint64_t m_nMatch;
    uint64_t m_nEmpty;
    uint64_t m_nTooOld;
    uint64_t m_nTimedOut;
    int _dataFd;
    char *_discard;
    // out-of-order support
    unsigned m_count;
    unsigned m_countOffset;
    bool m_resetHwCount;
    bool m_outOfOrder;
    ZmqContext m_context;
    ZmqSocket m_notifySocket;
};


class UdpApp : public CollectionApp
{
public:
    UdpApp(Parameters& para, std::shared_ptr<UdpMonitor> udpMonitor);
    ~UdpApp();
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
    void _shutdown();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);
private:
    DrpBase m_drp;
    Parameters& m_para;
    std::unique_ptr<UdpDetector> m_udpDetector;
    Detector* m_det;
    bool m_unconfigure;
};

}

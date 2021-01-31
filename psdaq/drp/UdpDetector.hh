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
#if 0
    std::vector<uint32_t> _getDimensions(size_t length);
    std::vector<uint32_t> getData(void* data, size_t& length);
#endif /* 0 */
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
    void process(const XtcData::TimeStamp&);
    void addNames(unsigned segment, XtcData::Xtc& xtc);
    int drainFd(int fd);
    int reset();
    enum { DefaultDataPort = 5006 };
private:
    int _readData(uint32_t *encoderValue, uint16_t *frameCount);
    void _loopbackInit();
    void _loopbackFini();
    void _loopbackSend();
    void _worker();
    void _udpReceiver();
    void _timeout(const XtcData::TimeStamp& timestamp);
    void _matchUp();
    void _handleMatch(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg);
    void _handleYounger(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg);
    void _handleOlder(const XtcData::Dgram& pvDg, Pds::EbDgram& pgpDg);
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
    uint32_t m_firstDimKw;              // Revisit: Hack!
    int _dataFd;
    char *_discard;
    // out-of-order support
    unsigned _count;
    unsigned _countOffset;
    bool _resetHwCount;
    bool _outOfOrder;
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


// encoder header: 32 bytes
typedef struct {
    uint16_t    frameCount;         // network byte ordrer
    char        reserved1[2];
    char        version[4];
    char        hardwareID[16];
    char        reserved2;
    char        channelMask;
    char        errorMask;
    char        mode;
    char        reserved3[4];
} encoder_header_t;

static_assert(sizeof(encoder_header_t) == 32, "Data structure encoder_header_t is not size 32");


// encoder channel: 32 bytes
typedef struct {
    uint32_t    encoderValue;       // network byte ordrer
    char        reserved1[4];
    char        hardwareID[16];
    char        reserved2;
    char        channel;
    char        error;
    char        mode;
    char        reserved3[4];
} encoder_channel_t;

static_assert(sizeof(encoder_channel_t) == 32, "Data structure encoder_channel_t is not size 32");

}

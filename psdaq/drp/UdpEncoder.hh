#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <assert.h>
#include <cstdint>
#include <vector>
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
    uint16_t    scaleDenom;         // network byte order
} encoder_channel_t;

static_assert(sizeof(encoder_channel_t) == 32, "Data structure encoder_channel_t is not size 32");


// encoder frame: 64 bytes
typedef struct {
    encoder_header_t    header;
    encoder_channel_t   channel[1];
} encoder_frame_t;

static_assert(sizeof(encoder_frame_t) == 64, "Data structure encoder_frame_t is not size 64");
class UdpEncoder;

class UdpReceiver
{
public:
    UdpReceiver(const Parameters&           para,
                SPSCQueue<XtcData::Dgram*>& encQueue,
                SPSCQueue<XtcData::Dgram*>& bufferFreeList);
    ~UdpReceiver();
public:
    const std::string name() const { return "encoder"; }    // FIXME
public:
    void start();
    void stop();
    void setOutOfOrder(std::string errMsg);
    bool getOutOfOrder() { return (m_outOfOrder); }
    void setMissingData(std::string errMsg);
    bool getMissingData() { return (m_missingData); }
    void process();
    void loopbackSend();
    int drainDataFd();
    int reset();
    uint64_t nUpdates() { return m_nUpdates; }
    uint64_t nMissed() { return m_nMissed; }
private:
    void _read(XtcData::Dgram& dgram);
    int _readFrame(encoder_frame_t *frame, bool& missing);
    int _junkFrame();
    void _loopbackInit();
    void _loopbackFini();
    void _udpReceiver();
private:
    const Parameters&           m_para;
    SPSCQueue<XtcData::Dgram*>& m_encQueue;
    SPSCQueue<XtcData::Dgram*>& m_bufferFreelist;
    std::atomic<bool>           m_terminate;
    std::thread                 m_udpReceiverThread;
    int                         m_loopbackFd;
    struct sockaddr_in          m_loopbackAddr;
    uint16_t                    m_loopbackFrameCount;
    int                          _dataFd;
    // out-of-order support
    unsigned                    m_count;
    unsigned                    m_countOffset;
    bool                        m_resetHwCount;
    bool                        m_outOfOrder;
    bool                        m_missingData;
    ZmqContext                  m_context;
    ZmqSocket                   m_notifySocket;
    uint64_t                    m_nUpdates;
    uint64_t                    m_nMissed;
};


class Pgp : public PgpReader
{
public:
    Pgp(const Parameters& para, DrpBase& drp, Detector* det, const bool& running);
    Pds::EbDgram* next(uint32_t& evtIndex);
    const uint64_t nDmaRet() { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    Detector* m_det;
    Pds::Eb::TebContributor& m_tebContributor;
    static const int MAX_RET_CNT_C = 100;
    const bool& m_running;
    int32_t m_available;
    int32_t m_current;
    unsigned m_nodeId;
    uint64_t m_nDmaRet;
};


class Interpolator
{
public:
  Interpolator(unsigned n, unsigned o) : _idx(0), _t(n), _v(n), _coeff(o+1)
  {
    // check to make sure inputs are correct
    assert(_t.size() == _v.size());
    assert(_t.size() >= _coeff.size());
  }
  ~Interpolator() {}

public:
  void reset() { _idx=0;  std::fill(_t.begin(), _t.end(), 0); }
  void update(XtcData::TimeStamp t, unsigned v);
  unsigned calculate(XtcData::TimeStamp t, XtcData::Damage& damage) const;

private:
  unsigned            _idx;
  std::vector<double> _t;
  std::vector<double> _v;
  std::vector<double> _coeff;
};


class UdpEncoder : public XpmDetector
{
public:
    UdpEncoder(Parameters& para, DrpBase& drp);
    unsigned connect(std::string& msg, unsigned slowGroup);
    unsigned disconnect();
  //    std::string sconfigure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd);
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override { /* unused */ }
    unsigned unconfigure();
    void addNames(unsigned segment, XtcData::Xtc& xtc, const void* bufEnd);
    int reset() { return m_udpReceiver ? m_udpReceiver->reset() : 0; }
    const PgpReader* pgp() { return &m_pgp; }
    enum { DefaultDataPort = 5006 };
    enum { MajorVersion = 3, MinorVersion = 0, MicroVersion = 0 };
private:
    void _event(XtcData::Dgram& dgram, const void* const bufEnd, const encoder_frame_t& frame, uint32_t *rawValue, uint32_t *interpolatedValue);
    void _worker();
    void _timeout(std::chrono::milliseconds timeout);
    void _process(Pds::EbDgram* dgram);
    void _handleTransition(uint32_t pebbleIdx, Pds::EbDgram* pebbleDg);
  //void _handleL1Accept(const XtcData::Dgram& encDg, Pds::EbDgram& pgpDg);
    void _handleL1Accept(Pds::EbDgram& pgpDg, const encoder_frame_t& frame, uint32_t *rawValue, uint32_t *interpolatedValue);
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    enum {RawNamesIndex = NamesIndex::BASE, InterpolatedNamesIndex};
    enum { DiscardBufSize = 10000 };
    DrpBase& m_drp;
    Pgp m_pgp;
    std::shared_ptr<UdpReceiver> m_udpReceiver;
    std::thread m_workerThread;
    Interpolator m_interpolator;
    bool m_interpolating;
    int m_slowGroup;
    SPSCQueue<uint32_t> m_evtQueue;
    SPSCQueue<XtcData::Dgram*> m_encQueue;
    SPSCQueue<XtcData::Dgram*> m_bufferFreelist;
    std::vector<uint8_t> m_buffer;
    std::atomic<bool> m_terminate;
    std::atomic<bool> m_running;
    std::shared_ptr<Pds::MetricExporter> m_exporter;
    uint64_t m_nEvents;
    uint64_t m_nMatch;
    uint64_t m_nEmpty;
    uint64_t m_nTooOld;
    uint64_t m_nTimedOut;
};


class UdpApp : public CollectionApp
{
public:
    UdpApp(Parameters& para);
    ~UdpApp();
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connectionShutdown() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    void _unconfigure();
    void _disconnect();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);
private:
    Parameters& m_para;
    MemPoolCpu m_pool;
    DrpBase m_drp;
    std::unique_ptr<UdpEncoder> m_udpDetector;
    Detector* m_det;
    bool m_unconfigure;
};

}

/**
 * @file
 * @brief UDP encoder DRP: encoder frame layouts, the UDP frame receiver, the PGP reader, a polynomial interpolator, and the UdpEncoder, UdpDrp and UdpApp classes.
 */
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

/** Socket receive buffer size (10000) requested by createUdpSocket() in UdpEncoder.cc. */
#define UDP_RCVBUF_SIZE 10000

namespace Drp {

// encoder header: 32 bytes
typedef struct {
    /** Frame counter, in network byte order on the wire; UdpReceiver converts it and uses it to detect missing and out-of-order frames. */
    uint16_t    frameCount;         // network byte order
    uint8_t     reserved1[2];  ///< Two bytes named reserved1; not used in UdpEncoder.cc.
    /** Major version, in network byte order on the wire; the loopback sender writes UdpEncoder::MajorVersion. */
    uint16_t    majorVersion;       // network byte order
    uint8_t     minorVersion;  ///< Minor version; the loopback sender writes UdpEncoder::MinorVersion.
    uint8_t     microVersion;  ///< Micro version; the loopback sender writes UdpEncoder::MicroVersion.
    char        hardwareID[16];  ///< Hardware ID text (16 bytes), logged at debug level; the loopback sender writes LOOPBACK SIM.
    uint8_t     reserved2;  ///< One byte named reserved2; not used in UdpEncoder.cc.
    uint8_t     channelMask;  ///< Channel mask; the loopback sender writes 0x01, otherwise not read in UdpEncoder.cc.
    uint8_t     errorMask;  ///< Error mask; not read in UdpEncoder.cc.
    uint8_t     mode;  ///< Header mode byte; not read in UdpEncoder.cc.
    uint8_t     reserved3[4];  ///< Four bytes named reserved3; not used in UdpEncoder.cc.
} encoder_header_t;  ///< 32-byte encoder frame header (size checked by a static_assert).

static_assert(sizeof(encoder_header_t) == 32, "Data structure encoder_header_t is not size 32");


// encoder channel: 32 bytes
typedef struct {
    /** Encoder value, in network byte order on the wire; converted by UdpReceiver and recorded as encoderValue. */
    uint32_t    encoderValue;       // network byte order
    /** Timing value, in network byte order on the wire; converted and recorded as timing. */
    uint32_t    timing;             // network byte order
    /** Scale, in network byte order on the wire; converted and recorded as scale. */
    uint16_t    scale;              // network byte order
    char        hardwareID[16];  ///< Channel hardware ID text (16 bytes); not read in UdpEncoder.cc.
    uint8_t     reserved1;  ///< One byte named reserved1; not used in UdpEncoder.cc.
    uint8_t     channel;  ///< Channel number; not read in UdpEncoder.cc.
    uint8_t     error;  ///< Error byte; recorded as error.
    uint8_t     mode;  ///< Mode byte; recorded as mode.
    /** Scale denominator, in network byte order on the wire; converted and recorded as scaleDenom. */
    uint16_t    scaleDenom;         // network byte order
} encoder_channel_t;  ///< 32-byte encoder channel record (size checked by a static_assert).

static_assert(sizeof(encoder_channel_t) == 32, "Data structure encoder_channel_t is not size 32");


// encoder frame: 64 bytes
typedef struct {
    encoder_header_t    header;  ///< Frame header.
    encoder_channel_t   channel[1];  ///< The single channel record of the frame (UdpEncoder.cc reads channel[0] only).
} encoder_frame_t;  ///< 64-byte encoder UDP frame: a header followed by one channel record.

static_assert(sizeof(encoder_frame_t) == 64, "Data structure encoder_frame_t is not size 64");
struct UdpParameters;

/** Receives encoder frames on a UDP socket in its own thread, stores each one as a Dgram in a queue, and reports missing and out-of-order frames to the collection server. */
class UdpReceiver
{
public:
  /** Size the frame queue and free list for nBuffers entries, connect the error PUSH socket to the collection server, and create the UDP socket on the loopback port if one is set, else on UdpEncoder::DefaultDataPort. */
  UdpReceiver(const UdpParameters& para, size_t nBuffers);
    /** Close the UDP socket if it was opened. */
    ~UdpReceiver();
public:
    /** Return the string encoder (the code marks this FIXME). */
    const std::string name() const { return "encoder"; }    // FIXME
public:
    /** Reset the frame counter, (re)initialize the queue and free list with one buffer per entry (a Dgram plus one frame), start the receiver thread and, when a loopback port is set, open the loopback socket. */
    void start();
    /** Stop and join the receiver thread and shut down the queues. */
    void stop();
    /** On the first call, set the out-of-order flag, log errMsg as critical and send it as an asynchronous error message; later calls do nothing. */
    void setOutOfOrder(std::string errMsg);
    /** Return the out-of-order flag. */
    bool getOutOfOrder() { return (m_outOfOrder); }
    /** On the first call, set the missing-data flag, log errMsg as critical and send it as an asynchronous error message; later calls do nothing. */
    void setMissingData(std::string errMsg);
    /** Return the missing-data flag. */
    bool getMissingData() { return (m_missingData); }
    /** Read one frame into a free buffer and push it onto the frame queue; if no buffer is free, count it as missed and discard the frame. */
    void process();
    /** Pop the oldest frame from the queue and return its buffer to the free list; returns false if the queue was empty. */
    bool consume();
    /** Consume all queued frames, printing a line for each. */
    void drain();
    /** Send one simulated frame (incremented frame count, encoder value 170000, timing 54321, scale 1 over 150, hardware ID LOOPBACK SIM) to the loopback port on 127.0.0.1. */
    void loopbackSend();
    /** Read and discard every frame waiting on the UDP socket (logging the count); returns the last recvfrom() result. */
    int drainDataFd();
    /** Call drainDataFd() if the socket is open and return its result, otherwise return -1. */
    int reset();
    /** Return the queue of received frames (each a Dgram holding one encoder_frame_t). */
    SPSCQueue<XtcData::Dgram*>& encQueue() { return m_encQueue; }
    /** Return the number of frames processed since the receiver thread started. */
    uint64_t nUpdates() { return m_nUpdates; }
    /** Return the number of frames dropped because no buffer was free. */
    uint64_t nMissed() { return m_nMissed; }
private:
    void _read(XtcData::Dgram& dgram);
    int _readFrame(encoder_frame_t *frame, bool& missing);
    int _junkFrame();
    void _loopbackInit();
    void _loopbackFini();
    void _udpReceiver();
private:
    const UdpParameters&        m_para;
    SPSCQueue<XtcData::Dgram*>  m_encQueue;
    SPSCQueue<XtcData::Dgram*>  m_bufferFreelist;
    std::vector<uint8_t>        m_buffer;
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


/** PgpReader for the UDP encoder DRP that turns complete PGP events into EbDgrams in the pebble. */
class Pgp : public PgpReader
{
public:
    /** Construct the PgpReader (at most 100 buffers per read, 32 per free batch) and set the driver mask for laneMask on the virtual channel of det, logging an error on failure. */
    Pgp(const UdpParameters& para, MemPool& pool, Detector* det);
    /** Return the next complete event as an EbDgram built in its pebble buffer (its index is stored in evtIndex), reading more DMA buffers when the previous batch is used up. Returns nullptr if nothing was read or the buffer did not complete an event. */
    Pds::EbDgram* next(uint32_t& evtIndex);
    /** Return the number of DMA buffers returned by the most recent read. */
    const uint64_t nDmaRet() { return m_nDmaRet; }
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex);
    Detector* m_det;
    static const unsigned MAX_RET_CNT_C = 100;
    int32_t m_available;
    int32_t m_current;
    uint64_t m_nDmaRet;
};


/** Least-squares polynomial fit (Eigen QR) over the last n (time, value) points, used to estimate encoder values at event times. */
class Interpolator
{
public:
  /** Keep n points and fit a polynomial of order o; asserts that n is at least o + 1. */
  Interpolator(unsigned n, unsigned o) : _idx(0), _t(n), _v(n), _coeff(o+1)
  {
    // check to make sure inputs are correct
    assert(_t.size() == _v.size());
    assert(_t.size() >= _coeff.size());
  }
  /** Does nothing; empty body. */
  ~Interpolator() {}

public:
  /** Set the write index to 0 and zero the stored times; the stored values are kept. */
  void reset() { _idx=0;  std::fill(_t.begin(), _t.end(), 0); }
  /** Store (t, v) in the circular buffer and, once all n slots hold a time, refit the polynomial with times taken relative to the oldest point. */
  void update(XtcData::TimeStamp t, unsigned v);
  /** Evaluate the fitted polynomial at ts once all slots are filled; otherwise return the most recently stored value and add MissingData to damage. */
  unsigned calculate(XtcData::TimeStamp t, XtcData::Damage& damage) const;

private:
  unsigned            _idx;
  std::vector<double> _t;
  std::vector<double> _v;
  std::vector<double> _coeff;
};


/** XpmDetector for a UDP encoder: owns the UdpReceiver and records each frame as raw and, when interpolating, interpolated encoder data. */
class UdpEncoder : public XpmDetector
{
public:
    /** Construct the XpmDetector base and set virtual channel 0. */
    UdpEncoder(UdpParameters& para, MemPoolCpu& pool);
    /** Call XpmDetector::connect() and create the UdpReceiver with one buffer per pebble buffer. Returns 0, or 1 with errorMsg set if creation threw a std::string. */
    unsigned connect(const nlohmann::json& msg, const std::string& id, std::string& errorMsg);
    /** Call XpmDetector::shutdown() and release the UdpReceiver; returns 0. */
    unsigned disconnect();
  //    std::string sconfigure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd);
    /** Call XpmDetector::configure() (returning 1 if it fails) and declare the encoder names with addNames(0, ...); returns 0. */
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    /** Clear the names lookup table; returns 0. */
    unsigned unconfigure();
    /** Add the frame fields (frame count, timing, scale, scale denominator, mode, error) together with *interpolatedValue as interpolated data and with *rawValue as raw data; each part is skipped when its pointer is null. */
    void event_(XtcData::Dgram& dgram, const void* const bufEnd, const encoder_frame_t& frame, uint32_t *rawValue, uint32_t *interpolatedValue);
    /** Does nothing (the code comment says unused); events are recorded through event_(). */
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override { /* unused */ }
public:
    /** Declare the raw encoder names (algorithm raw, version MajorVersion.MinorVersion.MicroVersion) for segment, plus the interpolated names when interpolation is enabled. */
    void addNames(unsigned segment, XtcData::Xtc& xtc, const void* bufEnd);
    /** Return the UdpReceiver; empty before connect() and after disconnect(). */
    const std::shared_ptr<UdpReceiver>& udpReceiver() const { return m_udpReceiver; }
    /** Call UdpReceiver::reset() if a receiver exists, otherwise return 0. */
    int reset() { return m_udpReceiver ? m_udpReceiver->reset() : 0; }
    /** Default UDP data port. */
    enum { DefaultDataPort = 5006  /**< UDP data port used when no loopback port is set (5006). */ };
    /** Version written into the encoder names and the loopback frames. */
    enum { MajorVersion = 3, /**< Major version (3). */ MinorVersion = 0, /**< Minor version (0). */ MicroVersion = 0  /**< Micro version (0). */ };
private:
    enum {RawNamesIndex = NamesIndex::BASE, InterpolatedNamesIndex};
    enum { DiscardBufSize = 10000 };
    std::shared_ptr<UdpReceiver> m_udpReceiver;
};


/** DrpBase for the UDP encoder: a worker thread matches PGP timing events with encoder frames (optionally interpolating) and sends the results to the TEB. */
class UdpDrp : public DrpBase
{
public:
    /** Construct DrpBase and the PGP reader, install a TebReceiver, and enable interpolation only if the slowGroup kwarg is a readout group from 0 to 7. */
    UdpDrp(UdpParameters&, MemPoolCpu&, UdpEncoder&, ZmqContext&);
    /** Does nothing; empty body. */
    virtual ~UdpDrp() {}
    /** Call DrpBase::connect(). If the encTprAlias kwarg is set, enable interpolation and take slowGroup from the readout group of the TPR entries of the connect message, stopping at the one whose alias matches. Returns an empty string or the DrpBase error. */
    std::string connect(const nlohmann::json& msg, size_t id);
    /** Call DrpBase::configure() and return its result (empty on success). */
    std::string configure(const nlohmann::json& msg);
    /** Call DrpBase::unconfigure(), then stop and join the worker thread; returns 0. */
    unsigned unconfigure();
    /** Call DrpBase::startup() and start the worker thread; returns an empty string or the DrpBase error. */
    std::string startup(XtcData::Xtc& xtc, const void* be);
private:
    int  _setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter);
    void _worker();
    void _timeout(std::chrono::milliseconds timeout);
    void _process(Pds::EbDgram* dgram);
    void _handleTransition(uint32_t pebbleIdx, Pds::EbDgram* pebbleDg);
  //void _handleL1Accept(const XtcData::Dgram& encDg, Pds::EbDgram& pgpDg);
    void _handleL1Accept(Pds::EbDgram& pgpDg, const encoder_frame_t& frame, uint32_t *rawValue, uint32_t *interpolatedValue);
    void _sendToTeb(const Pds::EbDgram& dgram, uint32_t index);
private:
    const UdpParameters& m_para;
    UdpEncoder& m_det;
    Pgp m_pgp;
    std::thread m_workerThread;
    Interpolator m_interpolator;
    SPSCQueue<uint32_t> m_evtQueue;
    std::atomic<bool> m_terminate;
    std::atomic<bool> m_running;
    uint64_t m_nEvents;
    uint64_t m_nMatch;
    uint64_t m_nEmpty;
    uint64_t m_nTooOld;
    uint64_t m_nTimedOut;
};


// Remove the unique_ptr member and declare m_det as a raw pointer.
/** CollectionApp of the UDP encoder DRP process: owns the MemPoolCpu, the UdpEncoder and the UdpDrp and handles the collection transitions. */
class UdpApp : public CollectionApp
{
public:
    /** Register with the collection as a drp, open the PGP pool, initialize Python, and create the UdpEncoder and the UdpDrp. */
    UdpApp(UdpParameters& para);
    /** Call handleReset() and finalize Python. */
    ~UdpApp();
    /** Unsubscribe from the partition, unconfigure, disconnect and shut down the connection. */
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connectionShutdown() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    std::string _endrun(const nlohmann::json& phase1Info);
    void _unconfigure();
    void _disconnect();
    void _error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg);
private:
    UdpParameters&              m_para;
    MemPoolCpu                  m_pool;
    std::unique_ptr<UdpEncoder> m_det;
    std::unique_ptr<UdpDrp>     m_drp;
    bool                        m_unconfigure;
    std::string                 m_lastKey;
};

}

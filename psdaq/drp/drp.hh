/**
 * @file
 * @brief Core DRP types: command-line Parameters, PGP event and DMA buffer records, the Pebble buffer allocation and the MemPool buffer pools.
 */
#pragma once

#include <thread>
#include <vector>
#include <cstdint>
#include <map>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <string>
#include "spscqueue.hh"

/** Maximum number of PGP lanes per event (8); size of PGPEvent::buffers. */
#define PGP_MAX_LANES 8

namespace Pds {
    class EbDgram;
};

namespace Drp {

/** Fixed NamesId index values used by the DRP. */
enum NamesIndex
{
   BASE         = 0,  ///< Value 0; BldDetector.cc uses it as its base names index.
   CHUNKINFO    = 252,  ///< Value 252; NamesId index of the chunk information written by DrpBase.cc.
   STEPINFO     = 253,  ///< Value 253; named for step information (no use found in psdaq/drp).
   OFFSETINFO   = 254,  ///< Value 254; NamesId index of the offset information written by TebReceiver.cc.
   RUNINFO      = 255,  ///< Value 255; NamesId index of the run information written by DrpBase.cc.
};

/** Settings of a DRP process, filled mostly from command-line options (drp.cc and the other DRP executables) and from the connect information. */
struct Parameters
{
    /** Set partition to -1u (unset), nworkers to 10, nCubeWorkers to 0, detSegment to 0, laneMask to 0x1, loopbackPort to 0 and verbose to 0. batchSize, rogMask, maxTrSize and cubeKeepRaw are not initialized here. */
    Parameters() :
        partition(-1u),
        nworkers(10),
        nCubeWorkers(0),
        detSegment(0),
        laneMask(0x1),
        loopbackPort(0),
        verbose(0)
    {
    }
    unsigned partition;  ///< Partition number (-p option in drp.cc; -1u means unset). DrpBase.cc also treats bit (1 << partition) of the readout groups as the common readout group.
    unsigned nworkers;  ///< Number of workers (-W option in drp.cc; default 10).
    unsigned nCubeWorkers;  ///< Number of cube workers (-Q option in drp.cc; default 0).
    unsigned batchSize;  ///< Batch size; drp.cc sets it to 32 (its comment says it must be a power of 2).
    unsigned detSegment;  ///< Detector segment number; drp.cc parses it from the _N suffix of alias.
    uint8_t laneMask;  ///< Bit mask of the PGP lanes in use (-l option in drp.cc, hexadecimal; default 0x1).
    std::string alias;  ///< Unique name of the form detName_N (-u option in drp.cc).
    std::string detName;  ///< Detector name; drp.cc takes it from alias without the _N suffix.
    std::string device;  ///< Device file to open, for example by MemPoolCpu (-d option in drp.cc).
    std::string outputDir;  ///< Output directory (-o option in drp.cc).
    std::string instrument;  ///< Instrument name with any :station suffix removed (-P option in drp.cc).
    std::string detType;  ///< Detector type string (-D option in drp.cc); drp.cc uses it to decide which kwargs are accepted.
    std::string serNo;  ///< Serial number string (-S option in drp.cc).
    std::string collectionHost;  ///< Host of the collection server (-C option in drp.cc).
    std::string prometheusDir;  ///< Prometheus configuration directory (-M option in drp.cc).
    std::map<std::string,std::string> kwargs;  ///< Keyword arguments parsed from the -k options.
    uint32_t rogMask;  ///< Readout group mask built by DrpBase.cc from the connect information (0x00ff0000 plus one bit per readout group in use); SlowUpdates missing any of these groups are dropped.
    int loopbackPort;  ///< Loopback data port used by UdpEncoder when non-zero (default 0).
    unsigned verbose;  ///< Verbosity level (each -v in drp.cc adds one; default 0).
    size_t maxTrSize;  ///< Maximum transition datagram size in bytes; drp.cc sets 8 MiB and several other executables 256 KiB.
    bool   cubeKeepRaw;  ///< True when the -Q value in drp.cc is followed by a plus sign.
};

/** Size and driver index of one DMA buffer of an event. */
struct DmaBuffer
{
    int32_t size;  ///< Size returned by the DMA read for this buffer.
    uint32_t index;  ///< DMA buffer index (into MemPool::dmaBuffers).
};

/** One event being assembled from up to PGP_MAX_LANES lanes: a DmaBuffer per lane, the mask of lanes received and the pebble buffer index assigned when the event is complete. */
struct PGPEvent
{
    DmaBuffer buffers[PGP_MAX_LANES];  ///< DMA buffer of each lane.
    uint8_t mask = 0;  ///< Bit mask of the lanes received so far (0 when the entry is free).
    unsigned pebbleIndex;  ///< Pebble buffer index, allocated once all lanes in Parameters::laneMask have arrived.
};

/** One page-aligned allocation holding the L1Accept buffers followed by the transition buffers; each buffer ends with a sentinel word used to detect overruns. */
class Pebble
{
public:
    /** Release the allocation with delete if it is non-null. Note that create() allocates it with posix_memalign(). */
    ~Pebble() {
        if (m_buffer) {
            delete m_buffer;
            m_buffer = nullptr;
        }
    }
    /** Allocate nL1Buffers buffers of l1BufSize bytes and nTrBuffers buffers of trBufSize bytes (each size rounded up to 16 bytes, each pool rounded up to whole pages plus one extra page) and write sentinel words (0xcdcdcdcd for L1Accept, 0xefefefef for transitions) at the end of every buffer and after each pool. Throws a C string if the allocation fails. */
    void create(unsigned nL1Buffers, size_t l1BufSize, unsigned nTrBuffers, size_t trBufSize);

    /** Return the address of L1Accept buffer index. */
    inline uint8_t* operator [] (unsigned index) {
        uint64_t offset = index*m_bufferSize;
        return &m_buffer[offset];
    }
    /** Return the index of the L1Accept buffer that contains buffer. */
    inline unsigned index(void* buffer) const {
      return (static_cast<uint8_t*>(buffer) - m_buffer) / m_bufferSize;
    }
    /** Return the total allocation size in bytes. */
    size_t size() const {return m_size;}
    /** Return the start of the L1Accept buffer pool. */
    uint8_t* buffer() const { return m_buffer; }
    /** Return the size of one L1Accept buffer (rounded up to 16 bytes). */
    size_t bufferSize() const {return m_bufferSize;} // L1Accepts
    /** Return the size of one transition buffer (rounded up to 16 bytes). */
    size_t trBufSize()  const {return m_trBufSize;}
    /** Return the start of the transition buffer pool. */
    uint8_t* trBuffer() const { return m_trBuffer; }
    /** Return the number of transition buffers. */
    unsigned nTrBuffers() const { return m_nTrBuffers; }
    /** Return the number of L1Accept buffers. */
    unsigned nL1Buffers() const { return m_nL1Buffers; }
private:
    size_t   m_size;
    uint8_t* m_buffer;
    size_t   m_bufferSize;              // L11Accepts
    size_t   m_trBufSize;
    uint8_t* m_trBuffer;
    unsigned m_nTrBuffers;
    unsigned m_nL1Buffers;
};

/** Abstract set of buffer pools for one DRP: the DMA buffers (allocated by the driver; only counted here), the pebble buffers and the transition buffers. Subclasses open the device and provide fd(), setMaskBytes() and the release of DMA buffers. */
class MemPool
{
public:
    /** Create the transition buffer queue (TEB_TR_BUFFERS rounded up to a power of 2) and zero the counters. The subclass calls _initialize() once the DMA buffer count and size are known. */
    MemPool(const Parameters& para);
    /** Does nothing; empty body. */
    virtual ~MemPool() {};
    Pebble pebble;  ///< Pebble holding the L1Accept and transition buffers.
    std::vector<PGPEvent> pgpEvents;  ///< Events being assembled, indexed by event counter modulo nDmaBuffers().
    std::vector<Pds::EbDgram*> transitionDgrams;  ///< Transition datagram assigned to each pebble buffer index (set for non-L1Accept events).
    void** dmaBuffers;  ///< DMA buffer addresses mapped from the driver.
    /** Return the number of DMA buffers reported by the driver. */
    unsigned dmaCount() const {return m_dmaCount;}
    /** Return dmaCount() rounded up to a power of 2. */
    unsigned nDmaBuffers() const {return m_nDmaBuffers;}
    /** Return the size of one DMA buffer. */
    unsigned dmaSize() const {return m_dmaSize;}
    /** Return the number of pebble (L1Accept) buffers. */
    unsigned nbuffers() const {return m_nbuffers;}
    /** Return the size of one pebble L1Accept buffer. */
    size_t bufferSize() const {return pebble.bufferSize();}
    /** Pure virtual: return the device file descriptor. */
    virtual int fd() const = 0;
    /** Shut down the transition buffer queue and log how often DMA reads and returns failed. */
    void shutdown();
    /** Pop a free transition buffer as an EbDgram pointer, or return nullptr if none is available. */
    Pds::EbDgram* allocateTr();
    /** Check the overrun sentinels of transition buffer dgram (logging the first overrun), then push it back onto the free queue. */
    void freeTr(Pds::EbDgram* dgram);
    /** Count one DMA buffer allocation (the firmware does the actual allocation) and return the count before the increment. */
    unsigned allocateDma();
    /** Read up to count DMA buffers with dmaReadBulkIndex() on fd(), filling the per-buffer return sizes, indices, flags, errors and destinations. Returns its result; errors are logged (the first one at error level). */
    ssize_t readDma(uint32_t count,
                    int32_t* ret,
                    uint32_t* index,
                    uint32_t* flags,
                    uint32_t* errors,
                    uint32_t* dest);
    /** Return count DMA buffers (by index) to the driver through the subclass and log errors; on success the frees are counted. Returns the subclass result. */
    ssize_t freeDma(unsigned count, uint32_t* indices);
    /** Allocate the next pebble buffer index in order, blocking while all buffers are in use. Pebble buffers must be freed in allocation order. */
    unsigned allocate();
    /** Free the oldest pebble buffer (logging if the index is not the expected one), wake a blocked allocate() if the pool was full, and check that buffer's overrun sentinels. */
    void freePebble(unsigned);
    /** Free pebble buffers in order until none is in use. */
    void flushPebble();
    /** Return the number of DMA allocations minus frees. */
    int64_t dmaInUse() const { return m_dmaAllocs.load(std::memory_order_relaxed) -
                                      m_dmaFrees.load(std::memory_order_relaxed); }
    /** Return the number of pebble allocations minus frees. */
    int64_t inUse() const { return m_allocs.load(std::memory_order_relaxed) -
                                   m_frees.load(std::memory_order_relaxed); }
    /** Return guess_size() of the transition buffer queue. That queue holds the free buffers, so despite the name this is the approximate number of free transition buffers. */
    int64_t trInUse() const { return m_transitionBuffers.guess_size(); }
    /** Reset the DMA counters (only if no DMA buffer is in use, otherwise warn), the pebble counters (with a warning if buffers are still in use), the overrun flags and the error counts. */
    void resetCounters();
    /** Pure virtual: configure the driver to deliver the lanes in laneMask on virtual channel virtChan. */
    virtual int setMaskBytes(uint8_t laneMask, unsigned virtChan) = 0;
    /** Return this pool cast with static_cast to T*. */
    template <typename T> T* getAs() { return static_cast<T*>( this ); }
protected:
    void _initialize(const Parameters&);
private:
    virtual ssize_t _freeDma(unsigned count, uint32_t* indices) = 0;
protected:
    unsigned m_nDmaBuffers;             // Rounded up dmaCount
    unsigned m_nbuffers;
    unsigned m_dmaCount;
    unsigned m_dmaSize;
    SPSCQueue<void*> m_transitionBuffers;
    std::atomic<uint64_t> m_dmaAllocs;
    std::atomic<uint64_t> m_dmaFrees;
    std::atomic<uint64_t> m_allocs;
    std::atomic<uint64_t> m_frees;
    std::mutex m_lock;
    std::condition_variable m_condition;
    uint8_t m_dmaOverrun;
    uint8_t m_l1Overrun;
    uint8_t m_trOverrun;
    uint8_t m_pad;
    unsigned m_nDmaReadErr;
    unsigned m_nDmaRetErr;
};

/** MemPool for a PGP card accessed from the CPU: opens the device and maps its DMA buffers. */
class MemPoolCpu : public MemPool
{
public:
    /** Open para.device and map its DMA buffers (aborting on failure), write the 0xabababab sentinel at the end of each DMA buffer and initialize the base pool. */
    MemPoolCpu(const Parameters&);
    /** Unmap the DMA buffers and close the device. */
    virtual ~MemPoolCpu();
    /** Return the device file descriptor. */
    virtual int fd() const override {return m_fd;}
    /** On the first successful call, set the driver mask to deliver each lane in laneMask on virtual channel virtChan; later calls do nothing. Returns 0, or 1 if dmaSetMaskBytes() failed. */
    virtual int setMaskBytes(uint8_t laneMask, unsigned virtChan) override;
private:
    virtual ssize_t _freeDma(unsigned count, uint32_t* indices) override;
private:
    int m_fd;
    bool m_setMaskBytesDone;
};

}

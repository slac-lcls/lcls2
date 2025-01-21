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

#define PGP_MAX_LANES 8

namespace Pds {
    class EbDgram;
};

namespace Drp {

enum NamesIndex
{
   BASE         = 0,
   CHUNKINFO    = 252,
   STEPINFO     = 253,
   OFFSETINFO   = 254,
   RUNINFO      = 255,
};

struct Parameters
{
    Parameters() :
        partition(-1u),
        nworkers(10),
        detSegment(0),
        laneMask(0x1),
        loopbackPort(0),
        verbose(0)
    {
    }
    unsigned partition;
    unsigned nworkers;
    unsigned batchSize;
    unsigned detSegment;
    uint8_t laneMask;
    std::string alias;
    std::string detName;
    std::string device;
    std::string outputDir;
    std::string instrument;
    std::string detType;
    std::string serNo;
    std::string collectionHost;
    std::string prometheusDir;
    std::map<std::string,std::string> kwargs;
    uint32_t rogMask;
    int loopbackPort;
    unsigned verbose;
    size_t maxTrSize;
};

struct DmaBuffer
{
    int32_t size;
    uint32_t index;
};

struct PGPEvent
{
    DmaBuffer buffers[PGP_MAX_LANES];
    uint8_t mask = 0;
    unsigned pebbleIndex;
};

class Pebble
{
public:
    ~Pebble() {
        if (m_buffer) {
            delete m_buffer;
            m_buffer = nullptr;
        }
    }
    void create(unsigned nL1Buffers, size_t l1BufSize, unsigned nTrBuffers, size_t trBufSize);

    inline uint8_t* operator [] (unsigned index) {
        uint64_t offset = index*m_bufferSize;
        return &m_buffer[offset];
    }
    size_t size() const {return m_size;}
    size_t bufferSize() const {return m_bufferSize;}
private:
    size_t   m_size;
    size_t   m_bufferSize;
    uint8_t* m_buffer;
};

class MemPool
{
public:
    MemPool(const Parameters& para);
    virtual ~MemPool() {};
    Pebble pebble;
    std::vector<PGPEvent> pgpEvents;
    std::vector<Pds::EbDgram*> transitionDgrams;
    void** dmaBuffers;
    unsigned dmaCount() const {return m_dmaCount;}
    unsigned nDmaBuffers() const {return m_nDmaBuffers;}
    unsigned dmaSize() const {return m_dmaSize;}
    unsigned nbuffers() const {return m_nbuffers;}
    size_t bufferSize() const {return pebble.bufferSize();}
    virtual int fd() const = 0;
    void shutdown();
    Pds::EbDgram* allocateTr();
    void freeTr(Pds::EbDgram* dgram) { m_transitionBuffers.push(dgram); }
    unsigned allocateDma();
    unsigned allocate();
    void freeDma(unsigned count, uint32_t* indices);
    void freePebble();
    int64_t dmaInUse() const { return m_dmaAllocs.load(std::memory_order_relaxed) -
                                      m_dmaFrees.load(std::memory_order_relaxed); }
    int64_t inUse() const { return m_allocs.load(std::memory_order_relaxed) -
                                   m_frees.load(std::memory_order_relaxed); }
    void resetCounters();
    virtual int setMaskBytes(uint8_t laneMask, unsigned virtChan) = 0;
protected:
    void _initialize(const Parameters&);
private:
    virtual void _freeDma(unsigned count, uint32_t* indices) = 0;
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
};

class MemPoolCpu : public MemPool
{
public:
    MemPoolCpu(const Parameters&);
    virtual ~MemPoolCpu();
    virtual int fd() const override {return m_fd;}
    virtual int setMaskBytes(uint8_t laneMask, unsigned virtChan) override;
private:
    virtual void _freeDma(unsigned count, uint32_t* indices) override;
private:
    int m_fd;
    bool m_setMaskBytesDone;
};

}

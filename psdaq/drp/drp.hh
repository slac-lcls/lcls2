#pragma once

#include <thread>
#include <vector>
#include <cstdint>
#include <map>
#include <atomic>

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

struct DmaBuffer
{
    int32_t size;
    uint32_t index;
};

struct PGPEvent
{
    DmaBuffer buffers[PGP_MAX_LANES];
    uint8_t mask = 0;
    void* l3InpBuf;
    Pds::EbDgram* transitionDgram;
};

struct Parameters
{
    Parameters() :
        partition(-1),
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
    MemPool(Parameters& para);
    ~MemPool();
    Pebble pebble;
    std::vector<PGPEvent> pgpEvents;
    void** dmaBuffers;
    unsigned nbuffers() const {return m_nbuffers;}
    size_t bufferSize() const {return pebble.bufferSize();}
    unsigned dmaSize() const {return m_dmaSize;}
    int fd () const {return m_fd;}
    Pds::EbDgram* allocateTr();
    void shutdown();
    void freeTr(Pds::EbDgram* dgram) { m_transitionBuffers.push(dgram); }
    void allocate(unsigned count) { m_inUse.fetch_add(count, std::memory_order_acq_rel) ; }
    void release(unsigned count) { m_inUse.fetch_sub(count, std::memory_order_acq_rel); }
    const uint64_t inUse() const { return m_inUse.load(std::memory_order_relaxed); }
    int setMaskBytes(uint8_t laneMask, unsigned virtChan);
private:
    unsigned m_nbuffers;
    unsigned m_dmaSize;
    int m_fd;
    bool m_setMaskBytesDone;
    SPSCQueue<void*> m_transitionBuffers;
    std::atomic<unsigned> m_inUse;
};

}

#pragma once

#include <thread>
#include <vector>
#include <cstdint>
#include "spscqueue.hh"

namespace Drp {

const int MAX_RET_CNT_C = 1000;

struct DmaBuffer
{
    int32_t size;
    uint32_t index;
};

struct PGPEvent
{
    DmaBuffer buffers[4];
    uint8_t mask = 0;
};

struct Batch
{
    uint32_t start;
    uint32_t size;
};

struct Parameters
{
    unsigned partition;
    unsigned nworkers;
    unsigned batchSize;
    unsigned detSegment;
    uint8_t laneMask;
    std::string detName;
    std::string device;
    std::string outputDir;
    std::string detectorType;
    std::string collectionHost;
};

class Pebble
{
public:
    void resize(unsigned nbuffers, unsigned bufferSize)
    {
        m_bufferSize = bufferSize;
        uint64_t size = static_cast<uint64_t>(nbuffers)*static_cast<uint64_t>(bufferSize);
        m_buffer.resize(size);
    }

    inline uint8_t* operator [] (unsigned index) {
        uint64_t offset = static_cast<uint64_t>(index)*static_cast<uint64_t>(m_bufferSize);
        return &m_buffer[offset];
    }
    size_t size() {return m_buffer.size();}
private:
    unsigned m_bufferSize;
    std::vector<uint8_t> m_buffer;
};

struct MemPool
{
    MemPool (const Parameters& para);
    Pebble pebble;
    std::vector<PGPEvent> pgpEvents;
    std::vector<SPSCQueue<Batch> > workerInputQueues;
    std::vector<SPSCQueue<Batch> > workerOutputQueues;
    unsigned nbuffers;
    void** dmaBuffers;
    int fd;
};

}

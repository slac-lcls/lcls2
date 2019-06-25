#pragma once

#include <thread>
#include <vector>
#include <cstdint>
#include <map>

namespace Drp {

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

struct Parameters
{
    unsigned partition;
    unsigned nworkers;
    unsigned batchSize;
    unsigned detSegment;
    uint8_t laneMask;
    std::string alias;
    std::string detName;
    std::string device;
    std::string outputDir;
    std::string detectorType;
    std::string collectionHost;
    std::map<std::string,std::string> kwargs;
};

class Pebble
{
public:
    void resize(unsigned nbuffers, unsigned bufferSize)
    {
        m_bufferSize = static_cast<uint64_t>(bufferSize);
        uint64_t size = static_cast<uint64_t>(nbuffers)*m_bufferSize;
        m_buffer.resize(size);
    }

    inline uint8_t* operator [] (unsigned index) {
        uint64_t offset = static_cast<uint64_t>(index)*m_bufferSize;
        return &m_buffer[offset];
    }
    size_t size() const {return m_buffer.size();}
private:
    uint64_t m_bufferSize;
    std::vector<uint8_t> m_buffer;
};

class MemPool
{
public:
    MemPool (const Parameters& para);
    Pebble pebble;
    std::vector<PGPEvent> pgpEvents;
    void** dmaBuffers;
    unsigned nbuffers () const {return m_nbuffers;}
    int fd () const {return m_fd;}
private:
    unsigned m_nbuffers;
    int m_fd;
};

}

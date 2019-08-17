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
    void* l3InpBuf;
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
    uint16_t rogMask;
};

class Pebble
{
public:
    void resize(unsigned nbuffers, size_t bufferSize)
    {
        m_bufferSize = bufferSize;
        size_t size = nbuffers*m_bufferSize;
        m_buffer.resize(size);
    }

    inline uint8_t* operator [] (unsigned index) {
        uint64_t offset = index*m_bufferSize;
        return &m_buffer[offset];
    }
    size_t size() const {return m_buffer.size();}
    size_t bufferSize() const {return m_bufferSize;}
private:
    size_t m_bufferSize;
    std::vector<uint8_t> m_buffer;
};

class MemPool
{
public:
    MemPool(const Parameters& para);
    Pebble pebble;
    std::vector<PGPEvent> pgpEvents;
    void** dmaBuffers;
    unsigned nbuffers() const {return m_nbuffers;}
    unsigned bufferSize() const {return m_bufferSize;}
    unsigned dmaSize() const {return m_dmaSize;}
    int fd () const {return m_fd;}
private:
    unsigned m_nbuffers;
    unsigned m_bufferSize;
    unsigned m_dmaSize;
    int m_fd;
};

}

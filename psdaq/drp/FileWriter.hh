#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include "psdaq/service/Fifo.hh"
#include "psdaq/service/Task.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/TimeStamp.hh"

namespace Drp {

class BufferedFileWriter
{
public:
    BufferedFileWriter(size_t bufferSize);
    ~BufferedFileWriter();
    int open(const std::string& fileName);
    int close();
    void writeEvent(void* data, size_t size, XtcData::TimeStamp ts);
    const uint64_t& writing() const { return m_writing; }
private:
    int m_fd;
    size_t m_count;
    XtcData::TimeStamp m_batch_starttime;
    std::vector<uint8_t> m_buffer;
    uint64_t m_writing;
};

class BufferedFileWriterMT
{
public:
    BufferedFileWriterMT(size_t bufferSize);
    ~BufferedFileWriterMT();
    int open(const std::string& fileName);
    int close();
    void writeEvent(void* data, size_t size, XtcData::TimeStamp ts);
    void run();
    const uint64_t& depth() const { return m_depth; }
    const uint64_t& size()  const { return m_size; }
    const uint64_t& writing() const { return *const_cast<uint64_t*>(&m_writing); }
    const uint64_t& freeBlocked()  const { return *const_cast<uint64_t*>(&m_freeBlocked); }
    const uint64_t& pendBlocked()  const { return *const_cast<uint64_t*>(&m_pendBlocked); }
private:
    size_t m_bufferSize;
    int m_fd;
    XtcData::TimeStamp m_batch_starttime;
    class Buffer {
    public:
        uint8_t* p;
        size_t   count;
    };
    Pds::FifoW<Buffer> m_free;
    Pds::FifoW<Buffer> m_pend;
    uint64_t m_depth;
    uint64_t m_size;
    volatile uint64_t m_writing;
    volatile uint64_t m_freeBlocked;
    volatile uint64_t m_pendBlocked;
    std::atomic<bool> m_terminate;
    std::thread m_thread;
};

class SmdDef : public XtcData::VarDef
{
public:
    enum index {
        intOffset,
        intDgramSize
    };

    SmdDef()
    {
        NameVec.push_back({"intOffset", XtcData::Name::UINT64});
        NameVec.push_back({"intDgramSize", XtcData::Name::UINT64});
    }
};

class SmdWriter : public BufferedFileWriter
{
public:
    SmdWriter(size_t bufferSize);
    void addNames(XtcData::Xtc& parent, unsigned nodeId);
    uint8_t buffer[0x4000000];
    XtcData::NamesLookup namesLookup;
};

}

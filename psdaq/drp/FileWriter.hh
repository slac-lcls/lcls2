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

#include "FileWriterBase.hh"

namespace Drp {

class BufferedFileWriter : public FileWriterBase
{
public:
    BufferedFileWriter(size_t bufferSize);
    ~BufferedFileWriter() override;
    int open(const std::string& fileName) override;
    int close() override;
    void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) override;
private:
    int m_fd;
    size_t m_count;
    XtcData::TimeStamp m_batch_starttime;
    std::vector<uint8_t> m_buffer;
};

class BufferedFileWriterMT : public FileWriterBase
{
public:
    BufferedFileWriterMT(size_t bufferSize);
    BufferedFileWriterMT(size_t bufferSize, bool dio);
    ~BufferedFileWriterMT() override;
    int open(const std::string& fileName) override;
    int close() override;
    void flush();
    void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) override;
    void run();
    uint64_t depth() const { return m_depth; }
    uint64_t size()  const { return m_size; }
    uint64_t freeBlocked()  const { return m_freeBlocked; }
    uint64_t pendBlocked()  const { return m_pendBlocked; }
private:
    void _initialize(size_t bufferSize);
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
    volatile uint64_t m_freeBlocked;
    volatile uint64_t m_pendBlocked;
    std::atomic<bool> m_terminate;
    std::thread m_thread;
    bool m_dio;
};

class BufferedMultiFileWriterMT : public FileWriterBase
{
public:
    BufferedMultiFileWriterMT(size_t bufferSize, size_t numFiles);
    BufferedMultiFileWriterMT(size_t bufferSize, size_t numFiles, bool dio);
    ~BufferedMultiFileWriterMT() override;
    int open(const std::string& fileName) override;
    int close() override;
    void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) override;
    void run();
    using FileWriterBase::writing;
    uint64_t depth      (size_t i) const { return m_fileWriters[i]->depth(); }
    uint64_t size       (size_t i) const { return m_fileWriters[i]->size(); }
    uint64_t writing    (size_t i) const { return m_fileWriters[i]->writing(); }
    uint64_t freeBlocked(size_t i) const { return m_fileWriters[i]->freeBlocked(); }
    uint64_t pendBlocked(size_t i) const { return m_fileWriters[i]->pendBlocked(); }
private:
    std::vector< std::unique_ptr<BufferedFileWriterMT> > m_fileWriters;
    size_t m_index;
};

class SmdWriter : public SmdWriterBase
{
public:
    SmdWriter(size_t bufferSize, size_t maxTrSize);
    ~SmdWriter() override {}
    int open(const std::string& fileName) override { return m_fileWriter.open(fileName); }
    int close() override { return m_fileWriter.close(); }
    void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) override { m_fileWriter.writeEvent(data, size, ts); }
private:
    BufferedFileWriter m_fileWriter;
};

}

/**
 * @file
 * @brief Buffered file writers: single-threaded (BufferedFileWriter), with a writer thread (BufferedFileWriterMT), round-robin over several files (BufferedMultiFileWriterMT), and the small-data SmdWriter.
 */
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

/** Single-threaded file writer that collects events in one memory buffer and writes the buffer when the next event does not fit or the batch is more than 2 s old (by timestamp seconds). */
class BufferedFileWriter : public FileWriterBase
{
public:
    /** Allocate a buffer of bufferSize bytes; no file is open yet. */
    BufferedFileWriter(size_t bufferSize);
    /** Call close(). */
    ~BufferedFileWriter() override;
    /** Close any open file (with a warning), create or truncate fileName with read permission for owner and group, and wait for a write lock on it. Returns 0, or -1 if the file cannot be created or locked. */
    int open(const std::string& fileName) override;
    /** If a file is open, write the buffered bytes and close it. Returns the close() result, or 0 if no file was open. */
    int close() override;
    /** Append size bytes from data to the buffer. The buffer is written out first if data does not fit or the batch started more than 2 s earlier; a datagram larger than the free space is written through the buffer in chunks. Aborts if a write fails. */
    void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) override;
private:
    int m_fd;
    size_t m_count;
    XtcData::TimeStamp m_batch_starttime;
    std::vector<uint8_t> m_buffer;
};

/** File writer with a pool of page-aligned buffers and a writer thread: writeEvent() fills buffers and queues full ones, and the thread writes them to the file. */
class BufferedFileWriterMT : public FileWriterBase
{
public:
    /** Create 64 buffers of bufferSize bytes rounded up to whole pages and start the writer thread. */
    BufferedFileWriterMT(size_t bufferSize);
    /** Like BufferedFileWriterMT(bufferSize). With dio true there are 2 buffers, each the smallest multiple of bufferSize that is at least 512 MiB (then rounded up to whole pages), and files are opened with O_DIRECT. */
    BufferedFileWriterMT(size_t bufferSize, bool dio);
    /** Close the file, stop and join the writer thread, and free the buffers in the free queue. */
    ~BufferedFileWriterMT() override;
    /** Like BufferedFileWriter::open(), adding O_DIRECT when direct I/O was requested. */
    int open(const std::string& fileName) override;
    /** If a file is open, call flush() (which waits until all queued buffers are written) and close the file. Returns the close() result, or 0 if no file was open. */
    int close() override;
    /** Queue the current buffer if it holds data, then block until the writer thread has written everything queued. */
    void flush();
    /** Copy size bytes from data into the current free buffer, first queueing that buffer if data does not fit or the batch is more than 2 s old; a datagram larger than the free space is split across buffers. Blocks while no free buffer is available. */
    void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) override;
    /** Writer thread body started by the constructor: name the thread drp/FileWriter, write each queued buffer to the file and return it to the free queue, until terminated with nothing queued. Aborts if a write fails. */
    void run();
    /** Return the number of free buffers recorded at the last queue operation. */
    uint64_t depth() const { return m_depth; }
    /** Return the number of buffers in the pool. */
    uint64_t size()  const { return m_size; }
    /** Return the counter that is raised while writeEvent() waits for a free buffer. */
    uint64_t freeBlocked()  const { return m_freeBlocked; }
    /** Return the counter that is raised while flush() or the writer thread waits on the pending queue. */
    uint64_t pendBlocked()  const { return m_pendBlocked; }
private:
    void _initialize(size_t bufferSize);
private:
    size_t m_bufferSize;
    int m_fd;
    XtcData::TimeStamp m_batch_starttime;
    class Buffer {
    public:
        uint8_t* p;  ///< Start of the page-aligned buffer memory.
        size_t   count;  ///< Number of bytes filled in the buffer.
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

/** Writes events round-robin to several BufferedFileWriterMT files. */
class BufferedMultiFileWriterMT : public FileWriterBase
{
public:
    /** Create numFiles BufferedFileWriterMT writers with buffers of bufferSize bytes. */
    BufferedMultiFileWriterMT(size_t bufferSize, size_t numFiles);
    /** Create numFiles BufferedFileWriterMT writers with buffers of bufferSize bytes and the given direct-I/O setting. */
    BufferedMultiFileWriterMT(size_t bufferSize, size_t numFiles, bool dio);
    /** Empty body; the writers are destroyed with their vector. */
    ~BufferedMultiFileWriterMT() override;
    /** Open one file per writer, named from fileName by inserting -iNN (00, 01, ...) before the last dot. Returns -1 if fileName has no dot, otherwise the first non-zero open() result, or 0. */
    int open(const std::string& fileName) override;
    /** Close each writer, stopping at the first non-zero result, which is returned (0 if all succeed). */
    int close() override;
    /** Pass the event to the next writer in round-robin order. */
    void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) override;
    /** Declared here; no definition was found in psdaq. */
    void run();
    /** Brings FileWriterBase::writing() into scope next to the per-file writing(i) overload. */
    using FileWriterBase::writing;
    /** Return depth() of writer i. */
    uint64_t depth      (size_t i) const { return m_fileWriters[i]->depth(); }
    /** Return size() of writer i. */
    uint64_t size       (size_t i) const { return m_fileWriters[i]->size(); }
    /** Return writing() of writer i. */
    uint64_t writing    (size_t i) const { return m_fileWriters[i]->writing(); }
    /** Return freeBlocked() of writer i. */
    uint64_t freeBlocked(size_t i) const { return m_fileWriters[i]->freeBlocked(); }
    /** Return pendBlocked() of writer i. */
    uint64_t pendBlocked(size_t i) const { return m_fileWriters[i]->pendBlocked(); }
private:
    std::vector< std::unique_ptr<BufferedFileWriterMT> > m_fileWriters;
    size_t m_index;
};

/** Small-data writer that forwards open, close and writeEvent to an internal BufferedFileWriter. */
class SmdWriter : public SmdWriterBase
{
public:
    /** Create the scratch buffer of maxTrSize bytes and an internal BufferedFileWriter with bufferSize bytes. */
    SmdWriter(size_t bufferSize, size_t maxTrSize);
    /** Does nothing; empty body. */
    ~SmdWriter() override {}
    /** Open fileName with the internal BufferedFileWriter and return its result. */
    int open(const std::string& fileName) override { return m_fileWriter.open(fileName); }
    /** Close the internal BufferedFileWriter and return its result. */
    int close() override { return m_fileWriter.close(); }
    /** Pass the event to the internal BufferedFileWriter. */
    void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) override { m_fileWriter.writeEvent(data, size, ts); }
private:
    BufferedFileWriter m_fileWriter;
};

}

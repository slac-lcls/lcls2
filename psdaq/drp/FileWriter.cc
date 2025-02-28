#include <errno.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include "FileWriter.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;


namespace Drp {

static inline ssize_t _write(int fd, const void* buffer, size_t count)
{
    auto sz = write(fd, buffer, count);

    while (size_t(sz) != count) {
        if (sz < 0) {
            // %m will be replaced by the string strerror(errno)
            logging::error("write error: %m");
            return sz;
        }
        count -= sz;
        sz = write(fd, (uint8_t*)buffer + sz, count);
    }
    return sz;
}


BufferedFileWriter::BufferedFileWriter(size_t bufferSize) :
    m_count(0), m_batch_starttime(0,0), m_buffer(bufferSize), m_writing(0)
{
}

BufferedFileWriter::~BufferedFileWriter()
{
    m_writing += 2;
    _write(m_fd, m_buffer.data(), m_count);
    m_writing -= 2;
    m_count = 0;
}

int BufferedFileWriter::open(const std::string& fileName)
{
    int rv = -1;
    struct flock flk;
    flk.l_type   = F_WRLCK;
    flk.l_whence = SEEK_SET;
    flk.l_start  = 0;
    flk.l_len    = 0;

    m_fd = ::open(fileName.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IRGRP);
    if (m_fd == -1) {
        // %m will be replaced by the string strerror(errno)
        logging::error("Error creating file %s: %m", fileName.c_str());
    } else {
        // establish write lock on the file
        int rc;
        do {
            rc = fcntl(m_fd, F_SETLKW, &flk);
        } while (rc<0 && errno==EINTR);
        if (rc<0) {
            // %m will be replaced by the string strerror(errno)
            logging::error("Error locking file %s: %m", fileName.c_str());
        } else {
            rv = 0;     // return OK
        }
    }
    return rv;
}

int BufferedFileWriter::close()
{
    int rv = 0;
    if (m_fd > 0) {
        if (m_count > 0) {
            logging::debug("Flushing %zu bytes to fd %d", m_count, m_fd);
            _write(m_fd, m_buffer.data(), m_count);
            m_count = 0;
            m_batch_starttime = XtcData::TimeStamp(0,0);
        }
        logging::debug("Closing fd %d", m_fd);
        rv = ::close(m_fd);
    } else {
        logging::warning("No file to close (m_fd=%d)", m_fd);
    }
    if (rv == -1) {
        // %m will be replaced by the string strerror(errno)
        logging::error("Error closing fd %d: %m", m_fd);
    } else {
        m_fd = 0;
    }
    return rv;
}

void BufferedFileWriter::writeEvent(const void* data, size_t size, XtcData::TimeStamp timestamp)
{
    // cpo: uncomment these two lines to get "unbuffered" writing
    // _write(m_fd, data, size);
    // return;

    // triggered only when starting from scratch
    if (m_batch_starttime.value()==0) m_batch_starttime = timestamp;

    // rough calculation: ignore nanoseconds
    unsigned age_seconds = timestamp.seconds()-m_batch_starttime.seconds();
    // write out data if buffer full or batch is too old
    // can't be 1 second without a more precise age calculation, since
    // the seconds field could have "rolled over" since the last event
    if ((size > (m_buffer.size() - m_count)) || age_seconds>2) {
        m_writing += 1;
        if (_write(m_fd, m_buffer.data(), m_count) == -1) {
            logging::critical("File writing failed");
            throw "File writing failed";
        }
        m_writing -= 1;
        // reset these to prepare for the new batch
        m_count = 0;
        m_batch_starttime = timestamp;
    }

    if (size>(m_buffer.size() - m_count)) {
        std::cout<<"Buffer size "<<(m_buffer.size()-m_count)<<" too small for dgram with size "<<size<<'\n';
        throw "FileWriter.cc buffer size too small";
    }
    memcpy(m_buffer.data()+m_count, data, size); // test if copy is slow
    m_count += size;
}

static const unsigned FIFO_DEPTH     = 64;
static const unsigned FIFO_DEPTH_DIO = 2;
static const size_t   FIFO_MIN_SIZE  = 512 * 1024 * 1024;

static size_t roundUpSize(size_t bufSize, size_t quantum)
{
    return quantum * ((bufSize + quantum - 1) / quantum);
}

BufferedFileWriterMT::BufferedFileWriterMT(size_t bufferSize) :
    m_fd(0),
    m_batch_starttime(0,0),
    m_free(FIFO_DEPTH),
    m_pend(FIFO_DEPTH),
    m_depth(m_free.size()),
    m_size(m_free.size()),
    m_writing(0),
    m_freeBlocked(0),
    m_pendBlocked(0),
    m_terminate(false),
    m_thread{&BufferedFileWriterMT::run,this},
    m_dio(false)
{
    _initialize(bufferSize);
}

BufferedFileWriterMT::BufferedFileWriterMT(size_t bufferSize, bool dio) :
    m_fd(0),
    m_batch_starttime(0,0),
    m_free(dio ? FIFO_DEPTH_DIO : FIFO_DEPTH),
    m_pend(dio ? FIFO_DEPTH_DIO : FIFO_DEPTH),
    m_depth(m_free.size()),
    m_size(m_free.size()),
    m_writing(0),
    m_freeBlocked(0),
    m_pendBlocked(0),
    m_terminate(false),
    m_thread{&BufferedFileWriterMT::run,this},
    m_dio(dio)
{
    _initialize(bufferSize);
}

BufferedFileWriterMT::~BufferedFileWriterMT()
{
    m_terminate = true;
    m_thread.join();
    Buffer b;
    while(!m_free.empty()) {
        m_free.pop(b);
        free(b.p);
    }
    m_depth = m_free.count();
}

void BufferedFileWriterMT::_initialize(size_t bufferSize)
{
    Buffer b;
    b.count = 0;
    if (m_dio)  bufferSize = roundUpSize(FIFO_MIN_SIZE, bufferSize); // N buffers >= FIFO_MIN_SIZE
    m_bufferSize = roundUpSize(bufferSize, sysconf(_SC_PAGESIZE));   // N pages
    for (unsigned i=0; i<m_free.size(); i++) {
        if (posix_memalign((void**)&b.p, sysconf(_SC_PAGESIZE), m_bufferSize)) {
          logging::critical("BufferedFileWriterMT posix_memalign: %m");
          throw "BufferedFileWriterMT posix_memalign";
        }
        m_free.push(b);
    }
}

int BufferedFileWriterMT::open(const std::string& fileName)
{
    int rv = -1;
    struct flock flk;
    flk.l_type   = F_WRLCK;
    flk.l_whence = SEEK_SET;
    flk.l_start  = 0;
    flk.l_len    = 0;

    auto oFlags = O_WRONLY | O_CREAT | O_TRUNC;
    if (m_dio)  oFlags |= O_DIRECT;
    m_fd = ::open(fileName.c_str(), oFlags, S_IRUSR | S_IRGRP);
    if (m_fd == -1) {
        // %m will be replaced by the string strerror(errno)
        logging::error("Error creating file %s: %m", fileName.c_str());
    } else {
        // establish write lock on the file
        int rc;
        do {
            rc = fcntl(m_fd, F_SETLKW, &flk);
        } while (rc<0 && errno==EINTR);
        if (rc<0) {
            // %m will be replaced by the string strerror(errno)
            logging::error("Error locking file %s: %m", fileName.c_str());
        } else {
            rv = 0;     // return OK
        }
    }
    return rv;
}

int BufferedFileWriterMT::close()
{
    int rv = 0;
    if (m_fd > 0) {
        flush();
        logging::debug("Closing fd %d", m_fd);
        rv = ::close(m_fd);
    } else {
        logging::warning("No file to close (m_fd=%d)", m_fd);
    }
    if (rv == -1) {
        // %m will be replaced by the string strerror(errno)
        logging::error("Error closing fd %d: %m", m_fd);
    } else {
        m_fd = 0;
    }
    return rv;
}

void BufferedFileWriterMT::flush()
{
    if (!m_free.empty() && m_free.front().count > 0) {
        Buffer b;
        m_free.pop(b);
        logging::debug("Flushing %zu bytes to fd %d", b.count, m_fd);
        m_pend.push(b);
        m_depth = m_free.count();
        m_batch_starttime = XtcData::TimeStamp(0,0);
    }
    m_pendBlocked += 2;
    m_pend.pendn();  // block until writing complete
    m_pendBlocked -= 2;
}

void BufferedFileWriterMT::writeEvent(const void* data, size_t size, XtcData::TimeStamp timestamp)
{
    // cpo: uncomment these two lines to get "unbuffered" writing
    // _write(m_fd, data, size);
    // return;

    // triggered only when starting from scratch
    if (m_batch_starttime.value()==0) m_batch_starttime = timestamp;

    // rough calculation: ignore nanoseconds
    unsigned age_seconds = timestamp.seconds()-m_batch_starttime.seconds();
    // write out data if buffer full or batch is too old
    // can't be 1 second without a more precise age calculation, since
    // the seconds field could have "rolled over" since the last event
    m_freeBlocked += 1;
    m_free.pend();
    m_freeBlocked -= 1;
    if ((size > (m_bufferSize - m_free.front().count)) || age_seconds>2) {
        Buffer b = m_free.front();
        m_free.pop(b);
        m_pend.push(b);
        m_depth = m_free.count();
        // reset these to prepare for the new batch
        m_batch_starttime = timestamp;
        m_freeBlocked += 2;
        m_free.pend();
        m_freeBlocked -= 2;
    }

    Buffer& b = m_free.front();
    if (size>(m_bufferSize - b.count)) {
        std::cout<<"Buffer size "<<(m_bufferSize-b.count)<<" too small for dgram with size "<<size<<'\n';
        throw "FileWriterMT.cc buffer size too small";
    }
    memcpy(b.p+b.count, data, size);
    b.count += size;
}

void BufferedFileWriterMT::run()
{
    while (true) {
        std::chrono::milliseconds tmo{100};
        m_pendBlocked += 1;
        m_pend.pend(tmo);
        m_pendBlocked -= 1;
        if (m_pend.empty()) {
            if (m_terminate.load(std::memory_order_relaxed)) {
                break;
            }
            else
                continue;
        }
        Buffer& b = m_pend.front();
        m_writing += 1;
        if (_write(m_fd, b.p, b.count) == -1) {
            logging::critical("File writing failed MT");
            throw "File writing failed";
        }
        m_writing -= 1;
        m_pend.pop(b);
        b.count = 0;
        m_free.push(b);
        m_depth = m_free.count();
    }
}

BufferedMultiFileWriterMT::BufferedMultiFileWriterMT(size_t bufferSize,
                                                     size_t numFiles) :
    m_index(0)
{
    while (numFiles--) {
        m_fileWriters.push_back(std::make_unique<BufferedFileWriterMT>(bufferSize));
    }
}


BufferedMultiFileWriterMT::~BufferedMultiFileWriterMT()
{
}

int BufferedMultiFileWriterMT::open(const std::string& fileName)
{
  auto& fn(fileName);
  auto dot = fn.find_last_of(".");
  if (dot == std::string::npos)  {
      logging::error("No '.' found in file spec '%s'", fileName.c_str());
      return -1;
  }
  auto base(fn.substr(0, dot));
  auto ext(fn.substr(dot));
  int rv = -1;
  unsigned i = 0;

  for (auto& writer : m_fileWriters) {
      std::ostringstream ss;
      ss << base << "-i" << std::setfill('0') << std::setw(2) << i++ << ext;
      rv = writer->open(ss.str());
      if (rv)  break;
      logging::debug("Opened file '%s'", ss.str().c_str());
      ss.seekp(0).clear();
  }
  return rv;
}


int BufferedMultiFileWriterMT::close()
{
  int rv = 0;

  for (auto& writer : m_fileWriters) {
      rv = writer->close();
      if (rv)  break;
  }
  return rv;
}

void BufferedMultiFileWriterMT::writeEvent(const void* data, size_t size, XtcData::TimeStamp timestamp)
{
  m_fileWriters[m_index]->writeEvent(data, size, timestamp);
  m_index = (m_index + 1) % m_fileWriters.size();
}

SmdWriter::SmdWriter(size_t bufferSize) : BufferedFileWriter(bufferSize)
{
}

void SmdWriter::addNames(XtcData::Xtc& parent, const void* bufEnd, unsigned nodeId)
{
    XtcData::Alg alg("offsetAlg", 0, 0, 0);
    XtcData::NamesId namesId(nodeId, 0);
    XtcData::Names& offsetNames = *new(parent, bufEnd) XtcData::Names(bufEnd, "info", alg, "offset", "", namesId);
    SmdDef smdDef;
    offsetNames.add(parent, bufEnd, smdDef);
    namesLookup[namesId] = XtcData::NameIndex(offsetNames);
}

}

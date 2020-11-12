#include <errno.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include "FileWriter.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;


namespace Drp {

BufferedFileWriter::BufferedFileWriter(size_t bufferSize) :
    m_count(0), m_batch_starttime(0,0), m_buffer(bufferSize), m_writing(0)
{
}

BufferedFileWriter::~BufferedFileWriter()
{
    ++m_writing;
    write(m_fd, m_buffer.data(), m_count);
    --m_writing;
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
            write(m_fd, m_buffer.data(), m_count);
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

void BufferedFileWriter::writeEvent(void* data, size_t size, XtcData::TimeStamp timestamp)
{
    // cpo: uncomment these two lines to get "unbuffered" writing
    // write(m_fd, data, size);
    // return;

    // triggered only when starting from scratch
    if (m_batch_starttime.value()==0) m_batch_starttime = timestamp;

    // rough calculation: ignore nanoseconds
    unsigned age_seconds = timestamp.seconds()-m_batch_starttime.seconds();
    // write out data if buffer full or batch is too old
    // can't be 1 second without a more precise age calculation, since
    // the seconds field could have "rolled over" since the last event
    if ((size > (m_buffer.size() - m_count)) || age_seconds>2) {
        ++m_writing;
        if (write(m_fd, m_buffer.data(), m_count) == -1) {
            // %m will be replaced by the string strerror(errno)
            logging::error("write error: %m");
            throw std::string("File writing failed");
        }
        --m_writing;
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

static const unsigned FIFO_DEPTH = 64;

BufferedFileWriterMT::BufferedFileWriterMT(size_t bufferSize) :
    m_bufferSize(bufferSize),
    m_batch_starttime(0,0),
    m_free(FIFO_DEPTH),
    m_pend(FIFO_DEPTH),
    m_terminate(false),
    m_thread{&BufferedFileWriterMT::run,this}
{
    Buffer b;
    b.count = 0;
    for(unsigned i=0; i<FIFO_DEPTH; i++) {
        b.p = new uint8_t[bufferSize];
        m_free.push(b);
    }
    m_size  = m_free.size();
    m_depth = m_free.count();
    m_writing     = 0;
    m_freeBlocked = 0;
    m_pendBlocked = 0;
}

BufferedFileWriterMT::~BufferedFileWriterMT()
{
    m_terminate = true;
    m_thread.join();
    Buffer b;
    while(!m_free.empty()) {
        m_free.pop(b);
        delete[] b.p;
    }
    m_depth = m_free.count();
}

int BufferedFileWriterMT::open(const std::string& fileName)
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

int BufferedFileWriterMT::close()
{
    int rv = 0;
    if (m_fd > 0) {
        if (!m_free.empty() && m_free.front().count > 0) {
            Buffer b;
            m_free.pop(b);
            logging::debug("Flushing %zu bytes to fd %d", b.count, m_fd);
            m_pend.push(b);
            m_depth = m_free.count();
            m_batch_starttime = XtcData::TimeStamp(0,0);
        }
        ++m_pendBlocked;
        m_pend.pendn();  // block until writing complete
        --m_pendBlocked;
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

void BufferedFileWriterMT::writeEvent(void* data, size_t size, XtcData::TimeStamp timestamp)
{
    // cpo: uncomment these two lines to get "unbuffered" writing
    // write(m_fd, data, size);
    // return;

    // triggered only when starting from scratch
    if (m_batch_starttime.value()==0) m_batch_starttime = timestamp;

    // rough calculation: ignore nanoseconds
    unsigned age_seconds = timestamp.seconds()-m_batch_starttime.seconds();
    // write out data if buffer full or batch is too old
    // can't be 1 second without a more precise age calculation, since
    // the seconds field could have "rolled over" since the last event
    ++m_freeBlocked;
    m_free.pend();
    --m_freeBlocked;
    if ((size > (m_bufferSize - m_free.front().count)) || age_seconds>2) {
        Buffer b = m_free.front();
        m_free.pop(b);
        m_pend.push(b);
        m_depth = m_free.count();
        // reset these to prepare for the new batch
        m_batch_starttime = timestamp;
        ++m_freeBlocked;
        m_free.pend();
        --m_freeBlocked;
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
        ++m_pendBlocked;
        m_pend.pend(tmo);
        --m_pendBlocked;
        if (m_pend.empty()) {
            if (m_terminate.load(std::memory_order_relaxed)) {
                break;
            }
            else
                continue;
        }
        Buffer& b = m_pend.front();
        ++m_writing;
        if (write(m_fd, b.p, b.count) == -1) {
            // %m will be replaced by the string strerror(errno)
            logging::error("write error: %m");
            throw std::string("File writing failed");
        }
        --m_writing;
        m_pend.pop(b);
        b.count = 0;
        m_free.push(b);
        m_depth = m_free.count();
    }
}

SmdWriter::SmdWriter(size_t bufferSize) : BufferedFileWriter(bufferSize)
{
}

void SmdWriter::addNames(XtcData::Xtc& parent, unsigned nodeId)
{
    XtcData::Alg alg("offsetAlg", 0, 0, 0);
    XtcData::NamesId namesId(nodeId, 0);
    XtcData::Names& offsetNames = *new(parent) XtcData::Names("info", alg, "offset", "", namesId);
    SmdDef smdDef;
    offsetNames.add(parent, smdDef);
    namesLookup[namesId] = XtcData::NameIndex(offsetNames);
}

}

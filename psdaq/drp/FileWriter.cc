#include <errno.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include "FileWriter.hh"
#include "psdaq/service/SysLog.hh"

using logging = Pds::SysLog;

namespace Drp {

BufferedFileWriter::BufferedFileWriter(size_t bufferSize) :
    m_count(0), m_buffer(bufferSize)
{
}

BufferedFileWriter::~BufferedFileWriter()
{
    write(m_fd, m_buffer.data(), m_count);
    m_count = 0;
}

void BufferedFileWriter::open(const std::string& fileName)
{
    m_fd = ::open(fileName.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
    if (m_fd == -1) {
        // %m will be replaced by the string strerror(errno)
        logging::error("Error creating file %s: %m", fileName.c_str());
    }
}

void BufferedFileWriter::writeEvent(void* data, size_t size)
{
    // cpo: uncomment these two lines to get "unbuffered" writing
    // write(m_fd, data, size);
    // return;

    // doesn't fit into the remaing m_buffer
    if (size > (m_buffer.size() - m_count)) {
        if (write(m_fd, m_buffer.data(), m_count) == -1) {
            // %m will be replaced by the string strerror(errno)
            logging::error("write error: %m");
            throw std::string("File writing failed");
        }
        m_count = 0;
    }
    if (size>(m_buffer.size() - m_count)) {
        std::cout<<"Buffer size "<<(m_buffer.size()-m_count)<<" too small for dgram with size "<<size<<'\n';
        throw "FileWriter.cc buffer size too small";
    }
    memcpy(m_buffer.data()+m_count, data, size);
    m_count += size;
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

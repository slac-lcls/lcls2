#pragma once

#include <string>
#include <vector>
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"

namespace Drp {

class BufferedFileWriter
{
public:
    BufferedFileWriter(size_t bufferSize);
    ~BufferedFileWriter();
    void open(const std::string& fileName);
    int close();
    void writeEvent(void* data, size_t size);
private:
    int m_fd;
    int m_count;
    std::vector<uint8_t> m_buffer;
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

#pragma once

#include <string>
#include <vector>
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/TimeStamp.hh"

namespace Drp {

class FileWriterBase
{
public:
    FileWriterBase() : m_writing(0) {}
    virtual ~FileWriterBase() {}
    virtual int open(const std::string& fileName) = 0;
    virtual int close() = 0;
    virtual void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) = 0;
    virtual uint64_t writing() const { return m_writing; }
protected:
    volatile uint64_t m_writing;
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

class SmdWriterBase : public FileWriterBase
{
public:
    SmdWriterBase(size_t maxTrSize) : buffer(maxTrSize) {}
    virtual ~SmdWriterBase() {}
    void addNames(XtcData::Xtc& parent, const void* bufEnd, unsigned nodeId);
    std::vector<uint8_t> buffer;
    XtcData::NamesLookup namesLookup;
};

} // Drp

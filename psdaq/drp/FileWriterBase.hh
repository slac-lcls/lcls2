/**
 * @file
 * @brief FileWriterBase, the abstract file writer interface, plus SmdDef and SmdWriterBase for small-data (offset) files.
 */
#pragma once

#include <string>
#include <vector>
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/TimeStamp.hh"

namespace Drp {

/** Abstract file writer interface (open, close, writeEvent) with a writing counter that the implementations in FileWriter.cc raise while a write() call is in progress. */
class FileWriterBase
{
public:
    /** Start with the writing counter at 0. */
    FileWriterBase() : m_writing(0) {}
    /** Does nothing; empty body. */
    virtual ~FileWriterBase() {}
    /** Pure virtual: open fileName for writing; the implementations in FileWriter.cc return 0 on success. */
    virtual int open(const std::string& fileName) = 0;
    /** Pure virtual: flush buffered data and close the current file. */
    virtual int close() = 0;
    /** Pure virtual: write size bytes from data; the implementations use the seconds of ts to age their batches. */
    virtual void writeEvent(const void* data, size_t size, XtcData::TimeStamp ts) = 0;
    /** Return the writing counter (non-zero while an implementation is inside a write). */
    virtual uint64_t writing() const { return m_writing; }
protected:
    volatile uint64_t m_writing;
};

/** VarDef of the small-data offset record: two UINT64 fields, intOffset and intDgramSize. */
class SmdDef : public XtcData::VarDef
{
public:
    /** Positions of the SmdDef fields. */
    enum index {
        intOffset,  ///< Index 0: intOffset (UINT64).
        intDgramSize  ///< Index 1: intDgramSize (UINT64).
    };

    /** Add intOffset and intDgramSize (both UINT64) to NameVec. */
    SmdDef()
    {
        NameVec.push_back({"intOffset", XtcData::Name::UINT64});
        NameVec.push_back({"intDgramSize", XtcData::Name::UINT64});
    }
};

/** FileWriterBase for small-data files, with a scratch buffer of transition size and a NamesLookup. */
class SmdWriterBase : public FileWriterBase
{
public:
    /** Allocate buffer with maxTrSize bytes. */
    SmdWriterBase(size_t maxTrSize) : buffer(maxTrSize) {}
    /** Does nothing; empty body. */
    virtual ~SmdWriterBase() {}
    /** Declared here, but no definition is compiled: FileWriterBase.cc defines SmdWriter::addNames() with this signature and is not listed in drp/CMakeLists.txt. No caller was found in psdaq. */
    void addNames(XtcData::Xtc& parent, const void* bufEnd, unsigned nodeId);
    std::vector<uint8_t> buffer;  ///< Scratch buffer of maxTrSize bytes, sized by the constructor.
    XtcData::NamesLookup namesLookup;  ///< Names lookup table for the small-data datagrams.
};

} // Drp

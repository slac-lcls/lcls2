/**
 * @file
 * @brief EpixHRemu, an emulated ePixHR detector that replays raw frames read from an XTC file.
 */
#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <vector>
#include <cstdint>

namespace Drp {

/** XpmDetector that emulates an ePixHR: the constructor loads one raw frame per active lane from the XTC file given by the xtcfile kwarg, and each event records those frames as a UINT16 raw array. */
class EpixHRemu : public XpmDetector
{
public:
    /** Set a synthetic serial number from detSegment and, if the xtcfile kwarg is set, read from that file one undamaged L1Accept raw frame per active lane, skipping (lanes x detSegment) plus the l1aOffset kwarg L1Accepts first. Aborts if the file cannot be opened. */
    EpixHRemu(Parameters* para, MemPool* pool);
    /** Call XpmDetector::configure() (returning 1 if it fails) and declare the raw names (raw, UINT16 rank 2); returns 0. */
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    /** Log a message and return 0. */
    unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
    /** Build the raw array from the stored frames: for each of the elemRows rows and each active lane, copy elemRowSize pixels from that lane's frame numAsics times (each copy from the same source offset); shape is elemRows by (lanes * numAsics * elemRowSize). l1count is not used. */
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override;
public:
    /** Number of ASICs per lane in the emulated frame (4). */
    const unsigned numAsics    = 4;
    /** Rows per ASIC (144). */
    const unsigned elemRows    = 144;
    /** Pixels per ASIC row (192). */
    const unsigned elemRowSize = 192;
    /** Pixels per stored frame: numAsics * elemRows * elemRowSize. */
    const unsigned numElems    = numAsics * elemRows * elemRowSize;
private:
    enum {RawNamesIndex = NamesIndex::BASE, FexNamesIndex};
    std::vector<uint8_t> m_rawBuffer[PGP_MAX_LANES];
};

}

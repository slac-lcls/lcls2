/**
 * @file
 * @brief AreaDetector, a generic DRP detector that records the PGP lane payloads as a raw 16-bit image plus a small FEX array, with cube binning support.
 */
#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesLookup.hh"

namespace Drp {

/** XpmDetector that records the payloads of the active PGP lanes as a raw UINT16 image (with a UINT32 value), adds a fixed 3x3 FEX array, and can bin the raw image into a cube with placeholder pedestals and gains. */
class AreaDetector : public XpmDetector
{
public:
    /** Construct the XpmDetector base; no calibration constants are allocated yet. */
    AreaDetector(Parameters* para, MemPool* pool);
    /** Call XpmDetector::configure() (returning 1 if it fails), then declare the fex names (array_fex, UINT16, rank 2) and the raw names (value UINT32, array_raw UINT16 rank 2). Returns 0. */
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    /** (Re)allocate the pedestal and gain arrays, one double per 16-bit sample of all lanes (length from the sw_sim_length kwarg or the configured length), and fill them with 0.5 and 1.0. Returns 0. */
    unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
    // Avoid "overloaded virtual function "Drp::Detector::event" is only partially overridden" warning
    using Detector::event;
    /** Add a 3x3 FEX array with values i + j, and a raw entry: value 9 followed by the DMA payload of each active lane after its 32-byte header (or sw_sim_length 32-bit words per lane if that kwarg is set), shaped as lanes by 16-bit samples. l1count is not used. */
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override;
    // For binning into the cube
    /** Accumulate raw data into cube bin bin: the value entry (only for subIndex 0) or, for array_raw, gain * (raw - pedestal) for each sample. Returns the entry size in bytes; note that dst is advanced by bin times that byte count in units of doubles. Aborts on an unknown valueIndex. */
    virtual unsigned addToCube(unsigned rawDefIndex, unsigned valueIndex, unsigned subIndex, 
                               double* dst, unsigned bin, XtcData::DescData& rawData) override;
    /** Return 0; the lane-count variant is compiled only when USE_SUB_INDEX is defined. */
    virtual unsigned subIndices    () override;
    /** Return the raw names index (RawNamesIndex, value 0). */
    virtual unsigned rawNamesIndex () override { return RawNamesIndex; }
    /** Return the cube names index (CubeNamesIndex, value 2). */
    virtual unsigned cubeNamesIndex() override { return CubeNamesIndex; }
    //  uint16_t -> double
    /** Return five times the pebble buffer size (the code comment notes the uint16_t to double conversion). */
    virtual unsigned cubeBinBytes  () override { return m_pool->bufferSize()*5; }
    /** Return a one-element list holding the raw definition (value and array_raw). */
    virtual std::vector<XtcData::VarDef>& rawDef () override;
private:
    enum {RawNamesIndex = NamesIndex::BASE, FexNamesIndex, CubeNamesIndex};
    enum {Pedestals, Gains, NumConstants};
    std::vector<char*> m_constants;
};

}

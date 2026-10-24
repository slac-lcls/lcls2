/**
 * @file
 * @brief TimingSystem, the DRP detector class for timing-system data (configuration through Python, TimingDef event data, cube binning of event codes and inhibit counts).
 */
#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

#include <Python.h>

namespace Drp {
class PythonConfigScanner;

/** XpmDetector for timing-system data. Configuration comes from the Python module `psdaq.configdb.{detType}_config`; L1Accept data is the timing record that follows the TimingHeader in the DMA buffer, described with TimingDef. */
class TimingSystem : public XpmDetector
{
public:
    /** Construct the XpmDetector base, import the module `psdaq.configdb.{detType}_config` and create a PythonConfigScanner for it. Throws a C string if the import fails. */
    TimingSystem(Parameters* para, MemPool* pool);
    /** Delete the config scanner and release the module reference. */
    ~TimingSystem();
    /** Call XpmDetector::connect(), then call ts_connect() of the module with the connect JSON, which is kept for configure(). */
    void connect(const nlohmann::json& msg, const std::string& collectionId) override;
    /** Call XpmDetector::configure() (returning 1 if it fails), append the configuration returned by ts_config() of the module as XTC (also taking serNo from its detId:RO entry), and declare the names of the L1Accept timing data (TimingDef) and of the trigger info (one UINT32 data field). Returns 0. */
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    /** Pass scan_keys to PythonConfigScanner::configure() with the update names ID and return its result. */
    unsigned configureScan(const nlohmann::json& scan_keys, XtcData::Xtc& xtc, const void* bufEnd) override;
    /** Print stepInfo to stdout and return 0. */
    unsigned beginstep(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& stepInfo) override;
    /** Pass stepInfo to PythonConfigScanner::step() with the update names ID and return its result. */
    unsigned stepScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;
    /** Return true; per the code comment this makes the detector record scan information. */
    bool scanEnabled() override;
    /** Describe the timing record that follows the TimingHeader in the DMA buffer of the first lane set in event->mask (TimingDef::describeData()); l1count is not used. */
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override;
    /** Add the trigger result word (result.data()) as the triginfo data value. */
    void event(XtcData::Dgram& dgram, const void* bufEnd, const Pds::Eb::ResultDgram& result) override;
    // For binning into the cube
    /** Return the cube shape for raw definition 0: 288 event-code bins for valueIndex 0 or 8 inhibit counts for valueIndex 1. Aborts for any other combination. */
    XtcData::Shape shapeCube(unsigned rawDefIndex, unsigned valueIndex, XtcData::DescData& rawData) override;
    /** Accumulate into cube bin bin: valueIndex 0 counts each set event-code bit of the 18 sequence words into 288 slots, valueIndex 1 adds the 8 inhibit counts. Case 0 has no break and falls through into case 1. Returns arraySize * sizeof(double_t) as left by the last case run (8 for valueIndex 0 or 1, 0 otherwise); aborts if rawDefIndex is not 0. */
    unsigned addToCube(unsigned rawDefIndex, unsigned valueIndex, unsigned subIndex, 
                       double* dst, unsigned bin, XtcData::DescData& rawData) override;
    /** Return the names index of the L1Accept data (EventNamesIndex, value 1). */
    unsigned rawNamesIndex () override;
    /** Return the names index of the cube data (CubeNamesIndex, value 4). */
    unsigned cubeNamesIndex() override;
  //    unsigned cubeBinBytes  () override;
    /** Return a one-element list of raw definitions used for cube binning: eventcodes (UINT8 array) and inhibitCounts (UINT32 array). */
    std::vector<XtcData::VarDef>& rawDef() override;
    /** Return the pebble buffer size when Parameters::nCubeWorkers is 0, otherwise 0x40000. */
    unsigned maxMonBufSize () override;
private:
    void _addJson(XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesId& configNamesId, const std::string& config_alias);
    enum {ConfigNamesIndex = NamesIndex::BASE, EventNamesIndex, UpdateNamesIndex, TriggerNamesIndex, CubeNamesIndex};
    XtcData::NamesId     m_evtNamesId;
    std::string          m_connect_json;
    PyObject*            m_module;
    PythonConfigScanner* m_configScanner;
};

}

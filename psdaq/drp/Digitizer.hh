/**
 * @file
 * @brief Digitizer, the DRP detector class for the high-speed digitizer, configured through the Python module psdaq.configdb.hsd_config.
 */
#pragma once

#include <vector>
#include "drp.hh"
#include "Detector.hh"
#include "psdaq/service/Semaphore.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "psalg/alloc/Allocator.hh"
#include <Python.h>

namespace Drp {
    class PythonConfigScanner;

    /** Detector for the high-speed digitizer (detType hsd per drp.cc), configured through the Python module psdaq.configdb.hsd_config with the EPICS prefix from the hsd_epics_prefix kwarg. */
    class Digitizer : public Detector
    {
    public:
        /** Keep the hsd_epics_prefix kwarg and the strm_limit kwarg (default 1000000), import psdaq.configdb.hsd_config, call its hsd_init with the prefix and device, and create a PythonConfigScanner. Aborts on a Python error. */
        Digitizer(Parameters* para, MemPool* pool);
        /** Delete the config scanner and release the Python module. */
        ~Digitizer();
        /** Call hsd_connect in psdaq.configdb.hsd_config with msg as a JSON string, read paddr from the returned dict and return xpmInfo(paddr) (xpm_id and xpm_port). Logs a critical message and aborts if paddr is 0, 0xffffffff, or its low byte is above 15. */
        nlohmann::json connectionInfo(const nlohmann::json& msg) override;
        /** Keep the connect message as a JSON string (passed to hsd_config by configure()) and set the readout group from body.drp.collectionId.det_info.readout of the message. */
        void connect(const nlohmann::json&, const std::string& collectionId) override;
        /** Call hsd_config in the Python module with the connect JSON, EPICS prefix, config_alias, detector name, segment and readout group, translate the returned JSON to XTC and append it to xtc, then add the L1Accept names (eventHeader, plus chan00 because the lane mask is fixed at 1). Returns 0; throws a C-string error if the JSON-to-XTC translation returns 0 or more than its 1 MiB buffer. */
        unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
        /** Store the two TimingHeader words as eventHeader and copy each lane's payload (after the TimingHeader) into name index lane+1, then walk the stream headers of the payload. Damage is increased (UserDefined) when bit 31 of the second word (JESD status, per the code comment) is clear, when a lane's size is at or above the strm_limit (that lane is skipped), or when a stream header reports unlocked; a stream header reporting overflow is logged as critical and aborts. */
        void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count) override;
        /** Call hsd_unconfig in psdaq.configdb.hsd_config with the EPICS prefix; its return value is not checked. */
        void shutdown() override;

        /** Pass the scan keys in stepInfo to PythonConfigScanner::configure (which calls the Python scan-keys function for the detector type) with names index UpdateNamesIndex, and return its result. */
        unsigned configureScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;
        /** Pass stepInfo to PythonConfigScanner::step (which calls the Python update function for the detector type) with names index UpdateNamesIndex, and return its result. */
        unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;

    private:
        unsigned _addJson(XtcData::Xtc& xtc, const void* bufEnd, XtcData::NamesId& configNamesId, const std::string& config_alias);
        void _dump(const void*, unsigned);
    private:
        enum {ConfigNamesIndex = NamesIndex::BASE, EventNamesIndex, UpdateNamesIndex};
        unsigned             m_readoutGroup;
        XtcData::NamesId     m_evtNamesId;
        std::string          m_connect_json;
        std::string          m_epics_name;
        unsigned             m_strm_limit;
        Pds::Semaphore       m_strm_limit_sem;
        Heap                 m_allocator;
        PyObject*            m_module;        // python module
        PythonConfigScanner* m_configScanner;
        unsigned             m_paddr;
    };

}

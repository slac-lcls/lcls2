#pragma once

#include <vector>
#include "drp.hh"
#include "Detector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "psalg/alloc/Allocator.hh"
#include <Python.h>

namespace Drp {
    class PythonConfigScanner;

    class Digitizer : public Detector
    {
    public:
        Digitizer(Parameters* para, MemPool* pool);
        ~Digitizer();
        nlohmann::json connectionInfo() override;
        void connect(const nlohmann::json&, const std::string& collectionId) override;
        unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) override;
        void event(XtcData::Dgram& dgram, PGPEvent* event) override;
        void shutdown() override;

        unsigned configureScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc) override;
        unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc) override;

    private:
        unsigned _addJson(XtcData::Xtc& xtc, XtcData::NamesId& configNamesId, const std::string& config_alias);
    private:
        enum {ConfigNamesIndex = NamesIndex::BASE, EventNamesIndex, UpdateNamesIndex};
        unsigned             m_readoutGroup;
        XtcData::NamesId     m_evtNamesId;
        std::string          m_connect_json;
        std::string          m_epics_name;
        Heap                 m_allocator;
        PyObject*            m_module;        // python module
        PythonConfigScanner* m_configScanner;
        unsigned             m_paddr;
    };

}

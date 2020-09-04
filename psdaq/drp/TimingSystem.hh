#pragma once

#include "drp.hh"
#include "XpmDetector.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

#include <Python.h>

namespace Drp {
class PythonConfigScanner;

class TimingSystem : public XpmDetector
{
public:
    TimingSystem(Parameters* para, MemPool* pool);
    ~TimingSystem();
    void connect(const nlohmann::json& msg, const std::string& collectionId) override;
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) override;
    unsigned configureScan(const nlohmann::json& scan_keys, XtcData::Xtc& xtc) override;
    unsigned beginstep(XtcData::Xtc& xtc, const nlohmann::json& stepInfo) override;
    unsigned stepScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc) override;
    bool scanEnabled() override;
    void event(XtcData::Dgram& dgram, PGPEvent* event) override;
private:
    void _addJson(XtcData::Xtc& xtc, XtcData::NamesId& configNamesId, const std::string& config_alias);
    enum {ConfigNamesIndex = NamesIndex::BASE, EventNamesIndex, UpdateNamesIndex};
    XtcData::NamesId     m_evtNamesId;
    std::string          m_connect_json;
    PyObject*            m_module;
    PythonConfigScanner* m_configScanner;
};

}

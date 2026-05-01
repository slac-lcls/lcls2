#pragma once

#include "Detector.hh"
#include <Python.h>

namespace XtcData {
    class Xtc;
    class Dgram;
}

namespace Drp {

class Parameters;
class MemPool;

class XpmDetector : public Detector
{
protected:
    XpmDetector(Parameters* para, MemPool* pool, unsigned len=100);
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    void connect(const nlohmann::json&, const std::string& collectionId) override;
    unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
    void shutdown() override;
    void connectionShutdown() override;
private:
    void _init();
private:
    PyObject*            m_xmodule;        // python module
protected:
    unsigned             m_length;
};

}

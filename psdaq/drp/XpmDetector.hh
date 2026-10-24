/**
 * @file
 * @brief XpmDetector, the Detector base for PGP-card devices that take their timing from an XPM link.
 */
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

/** Detector base for devices read through the PGP card with XPM timing. The constructor runs xpmdet_init from the Python module psdaq.configdb.xpmdet_config; connectionInfo() reports the XPM link from xpmdet_connectionInfo, connect() passes the readout group and event length (sim_length kwarg, else the constructor value, default 100) to xpmdet_connect, and configure() adds nothing and returns 0. */
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

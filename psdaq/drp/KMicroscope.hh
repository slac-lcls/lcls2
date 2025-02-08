#pragma once

#include "XpmDetector.hh"
#include "DrpBase.hh"
#include "PipeCallbackHandler.hh"
#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "BldDetector.hh" // Include the base BldApp class
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <queue>
#include <mutex>

namespace Drp {

// Detector class for KMicroscope
class KMicroscope : public XpmDetector {
public:
    KMicroscope(Parameters& para, DrpBase& drp);
    ~KMicroscope();

    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override;
};

/**
 * CustomBldApp is a specialized version of BldApp that initializes with
 * a KMicroscope detector instead of the default BldDetector.
 */
class CustomBldApp : public BldApp {
public:
    /**
     * Constructor now accepts:
     *  - para:       Application parameters.
     *  - drp:        A DrpBase object.
     *  - customParam: A custom parameter string for additional configuration.
     *  - measurementTimeMs: The measurement time in milliseconds.
     *  - iniFilePath: Path to the INI file for the KMicroscope detector.
     *  - batchSize:   (Optional) The batch size for event accumulation.
     */
    CustomBldApp(Parameters& para,
                 DrpBase& drp,
                 const std::string& customParam,
                 int measurementTimeMs,
                 const std::string& iniFilePath,
                 size_t batchSize = 100);

    ~CustomBldApp() override;

    void run();

private:
    std::string m_customParam;  // Custom parameter for additional configurations
    int m_measurementTimeMs;    // Measurement time in milliseconds
};

class KMicroscopeBld : public Bld {
public:
    // Constructor: Passes the measurement time (ms), INI file path, and optionally a batch size
    // to the PipeCallbackHandler.
    KMicroscopeBld(int measurementTimeMs,
                   const std::string& iniFilePath,
                   size_t batchSize = 100);

    // Destructor: The PipeCallbackHandler cleans up automatically.
    ~KMicroscopeBld();

    // Overrides Bld::next() to yield the next pulse ID (time_tag) from the PipeCallbackHandler.
    uint64_t next();

private:
    PipeCallbackHandler m_callbackHandler;  // Handles device communication and event collection.
    int m_measurementTimeMs;
};

} // namespace Drp

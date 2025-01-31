#pragma once

#include "XpmDetector.hh"
#include "DrpBase.hh"
#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "BldDetector.hh" // Include the base BldApp class
#include <stdio.h>
#include <stdlib.h>
#include <scTDC/scTDC.h>
#include <inttypes.h>

namespace Drp {

// Detector class for KMicroscope
class KMicroscope : public XpmDetector {
public:
    KMicroscope(Parameters& para, DrpBase& drp);
    ~KMicroscope();

    //unsigned configure(const std::string& configAlias, XtcData::Xtc& xtc, const void* bufEnd) override;
    void initialize();
    void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override;
    //void shutdown() override;
};

/**
 * CustomBldApp is a specialized version of BldApp that initializes with
 * a KMicroscope detector instead of the default BldDetector.
 */
class CustomBldApp : public BldApp {
public:
    // Constructor that initializes BldApp with KMicroscope
    CustomBldApp(Parameters& para, DrpBase& drp, const std::string& customParam);
    ~CustomBldApp() override;

    // Runs the application
    void run();

private:
    std::string m_customParam;  // Custom parameter for additional configurations
};

} // namespace Drp

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

// Derived App class for KMicroscope
class CustomBldApp : public BldApp {
public:
    CustomBldApp(Parameters& para, const std::string& customParam);
    ~CustomBldApp() override;
    void run();

private:
    std::string m_customParam;
};

} // namespace Drp

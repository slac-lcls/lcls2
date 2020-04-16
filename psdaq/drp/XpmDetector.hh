#pragma once

#include "Detector.hh"

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
    XpmDetector(Parameters* para, MemPool* pool);
    nlohmann::json connectionInfo() override;
    void connect(const nlohmann::json&, const std::string& collectionId) override;
protected:
    unsigned m_length;
};

}

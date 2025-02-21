#pragma once

#include "BEBDetector.hh"
#include "JungfrauDetectorId.hh"
#include "psalg/alloc/Allocator.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/Xtc.hh"

namespace sls
{
  class Detector;
} // namespace sls

namespace Drp
{

class JungfrauIdLookup;

class Jungfrau : public BEBDetector
{
public:
    Jungfrau(Parameters* para, MemPool* pool);
    virtual ~Jungfrau();

    void cleanup();

private:
    void _connectionInfo(PyObject*) override;
    unsigned _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void _event(XtcData::Xtc&, const void* bufEnd, std::vector<XtcData::Array<uint8_t>>&) override;
    bool _stopAcquisition();
    std::string _buildDetId(uint64_t sensor_id,
                            uint64_t board_id,
                            uint64_t firmware,
                            std::string software,
                            std::string hostname);

private:
    const unsigned m_nAsics = 1;
    const unsigned m_nRows = 512;
    const unsigned m_nCols = 1024;
    const unsigned m_nElems = m_nAsics * m_nRows * m_nCols;
    unsigned m_nModules = 0;
    std::vector<unsigned> m_segNos;
    std::vector<std::string> m_serNos;
    std::vector<std::string> m_slsHosts;
    std::unique_ptr<sls::Detector> m_slsDet;
    JungfrauIdLookup m_idLookup;
};

} // namespace Drp

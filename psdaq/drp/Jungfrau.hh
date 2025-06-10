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

    unsigned configureScan(const nlohmann::json& scanKeys, XtcData::Xtc& xtc, const void* bufEnd) override;
    unsigned stepScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;
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
    uint32_t _countNumHotPixels(uint16_t* rawData, uint16_t hotPixelThreshold, uint32_t numPixels);
    void _configure_module_thread(size_t mod,
                                  XtcData::Names& configNames,
                                  XtcData::ConfigIter& configo,
                                  std::atomic<unsigned>& numFailed);
    unsigned _configure_module(size_t mod, XtcData::Names& configNames, XtcData::ConfigIter& configo);

private:
    unsigned m_nModules { 0 };
    uint16_t m_hotPixelThreshold { 15000 };
    uint32_t m_maxHotPixels { 3400 };
    uint64_t m_expectedFrameNum { 0 };
    bool m_inFixedGain { false }; // Need to know if in fixed gain for hot pixel calc.
    std::vector<unsigned> m_segNos;
    std::vector<std::string> m_serNos;
    std::vector<std::string> m_slsHosts;
    std::unique_ptr<sls::Detector> m_slsDet;
    std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>> m_configEnums;
    JungfrauIdLookup m_idLookup;
};

} // namespace Drp

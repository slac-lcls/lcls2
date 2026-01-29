#pragma once

#include "drp.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/EbDgram.hh"

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>


namespace Pds {
  namespace Eb {
    class ResultDgram;
  }
}

namespace Drp {
    namespace Gpu {
        class Detector;
    }

struct Parameters;

class Detector
{
public:
    Detector(Parameters* para, MemPool* pool) :
        nodeId(-1u), virtChan(0), m_para(para), m_pool(pool), m_xtcbuf(para->maxTrSize) {}
    virtual ~Detector() {}

    virtual Gpu::Detector* gpuDetector() { return nullptr; }

    virtual nlohmann::json connectionInfo(const nlohmann::json& msg) {return nlohmann::json({});}
    virtual void connectionShutdown() {}
    virtual void connect(const nlohmann::json&, const std::string& collectionId) {};
    virtual unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) = 0;
    virtual unsigned beginrun (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) {return 0;}
    virtual unsigned beginstep(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& stepInfo) {return 0;};
    virtual unsigned enable   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) {return 0;};
    virtual unsigned disable  (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) {return 0;};
    virtual void slowupdate(XtcData::Xtc& xtc, const void* bufEnd) { xtc = {{XtcData::TypeId::Parent, 0}, {nodeId}}; };
    virtual void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) = 0;
    virtual void event(XtcData::Dgram& dgram, const void* bufEnd, const Pds::Eb::ResultDgram& result) {};
    virtual void cube(XtcData::Dgram& raw, unsigned binId, void* binXtc, unsigned& entries, const void* bufEnd) {};
    virtual void shutdown() {};

    // Scan methods.  Default is to fail.
    virtual unsigned configureScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) {return 1;};
    virtual unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) {return 1;};

    virtual Pds::TimingHeader* getTimingHeader(uint32_t index) const
    {
        return static_cast<Pds::TimingHeader*>(m_pool->dmaBuffers[index]);
    }
    virtual bool scanEnabled() {return false;}
    XtcData::Xtc& transitionXtc() {return *reinterpret_cast<XtcData::Xtc*>(m_xtcbuf.data());}
    const void*   trXtcBufEnd()   {return m_xtcbuf.data() + m_xtcbuf.size();}
    XtcData::NamesLookup& namesLookup() {return m_namesLookup;}
    unsigned nodeId;
    unsigned virtChan;
protected:
    Parameters* m_para;
    MemPool* m_pool;
    XtcData::NamesLookup m_namesLookup;
    std::vector<char> m_xtcbuf;
};

template <typename T>
class Factory
{
public:
    template <typename TDerived>
    void register_type(const std::string& name)
    {
        static_assert(std::is_base_of<T, TDerived>::value,
                      "Factory::register_type doesn't accept this type because doesn't derive from base class");
        m_create_funcs[name] = &createFunc<TDerived>;
    }

    T* create(Parameters* para, MemPool* pool)
    {
        std::string name = para->detType;
        auto it = m_create_funcs.find(name);
        if (it != m_create_funcs.end()) {
            return it->second(para, pool);
        }
        return nullptr;
    }

private:
    template <typename TDerived>
    static T* createFunc(Parameters* para, MemPool* pool)
    {
        return new TDerived(para, pool);
    }

    typedef T* (*PCreateFunc)(Parameters* para, MemPool* pool);
    std::unordered_map<std::string, PCreateFunc> m_create_funcs;
};

}

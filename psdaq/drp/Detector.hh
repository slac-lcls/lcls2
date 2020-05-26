#pragma once

#include "drp.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/EbDgram.hh"

#include <string>
#include <vector>
#include <unordered_map>
#include "psdaq/service/json.hpp"

namespace Drp {

struct Parameters;

class Detector
{
public:
    Detector(Parameters* para, MemPool* pool) :
        nodeId(-1), virtChan(0), m_para(para), m_pool(pool), m_xtcbuf(para->maxTrSize) {}
    virtual ~Detector() {}
    virtual nlohmann::json connectionInfo() {return nlohmann::json({});}
    virtual void connect(const nlohmann::json&, const std::string& collectionId) {};
    virtual unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc) = 0;
    virtual unsigned beginrun(XtcData::Xtc& xtc, const nlohmann::json& runInfo) {return 0;}
    virtual void beginstep(XtcData::Xtc& xtc, const nlohmann::json& stepInfo) {};
    virtual void slowupdate(XtcData::Xtc& xtc) { XtcData::Xtc& trXtc = transitionXtc();
                                                 memcpy(&xtc, &trXtc, trXtc.extent); };
    virtual void event(XtcData::Dgram& dgram, PGPEvent* event) = 0;
    virtual void shutdown() {};
    virtual Pds::TimingHeader* getTimingHeader(uint32_t index) const
    {
        return static_cast<Pds::TimingHeader*>(m_pool->dmaBuffers[index]);
    }
    XtcData::Xtc& transitionXtc() {return *reinterpret_cast<XtcData::Xtc*>(m_xtcbuf.data());}
    XtcData::NamesLookup& namesLookup() {return m_namesLookup;}
    unsigned nodeId;
    unsigned virtChan;
protected:
    Parameters* m_para;
    MemPool* m_pool;
    XtcData::NamesLookup m_namesLookup;
    std::vector<uint8_t> m_xtcbuf;
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

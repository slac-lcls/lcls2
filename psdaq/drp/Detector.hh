#pragma once

#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <string>
#include <vector>
#include <unordered_map>
#include "drp.hh"

namespace Drp {

struct Parameters;

class Detector
{
public:
    Detector(Parameters* para, MemPool* pool, unsigned nodeId) : m_para(para), m_pool(pool), m_nodeId(nodeId) {}
    virtual void connect() {};
    virtual unsigned configure(XtcData::Xtc& xtc) = 0;
    virtual void event(XtcData::Dgram& dgram, PGPEvent* event) = 0;
    XtcData::Xtc& transitionXtc() {return *(XtcData::Xtc*)m_xtcbuf;}
    unsigned nodeId() const {return m_nodeId;}
protected:
    Parameters* m_para;
    MemPool* m_pool;
    unsigned m_nodeId;
    std::vector<XtcData::NamesId> m_namesId;
    XtcData::NamesLookup m_namesLookup;
    uint8_t m_xtcbuf[XtcData::Dgram::MaxSize];
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

    T* create(Parameters* para, MemPool* pool, unsigned nodeId)
    {
        std::string name = para->detectorType;
        auto it = m_create_funcs.find(name);
        if (it != m_create_funcs.end()) {
            return it->second(para, pool, nodeId);
        }
        return nullptr;
    }

private:
    template <typename TDerived>
    static T* createFunc(Parameters* para, MemPool* pool, unsigned nodeId)
    {
        return new TDerived(para, pool, nodeId);
    }

    typedef T* (*PCreateFunc)(Parameters* para, MemPool* pool, unsigned nodeId);
    std::unordered_map<std::string, PCreateFunc> m_create_funcs;
};

}

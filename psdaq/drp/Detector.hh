#ifndef DETECTORS_H
#define DETECTORS_H

#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/NamesVec.hh"

#include <string>
#include <vector>
#include <unordered_map>

class Detector
{
public:
    virtual void configure(XtcData::Dgram& dgram, PGPData* pgp_data) = 0;
    virtual void event(XtcData::Dgram& dgram, PGPData* pgp_data) = 0;
    Detector(unsigned nodeId) : m_nodeId(nodeId) {}
protected:
    std::vector<XtcData::NamesId> m_namesId;
    XtcData::NamesVec             m_namesVec;
    unsigned                      m_nodeId;
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

    T* create(const std::string& name)
    {
        auto it = m_create_funcs.find(name);
        if (it != m_create_funcs.end()) {
            return it->second();
        }
        return nullptr;
    }

private:
    template <typename TDerived>
    static T* createFunc()
    {
        return new TDerived(0); // FIXME: kludged zero for the nodeId - cpo
    }

    typedef T* (*PCreateFunc)();
    std::unordered_map<std::string, PCreateFunc> m_create_funcs;
};

#endif // DETECTORS_H

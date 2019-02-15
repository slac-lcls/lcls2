#ifndef DETECTORS_H
#define DETECTORS_H

#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <string>
#include <vector>
#include <unordered_map>

class Detector
{
public:
    Detector(unsigned nodeId) : m_nodeId(nodeId) {}
    virtual void connect() {};
    virtual void configure(XtcData::Dgram& dgram, PGPData* pgp_data) = 0;
    virtual void event(XtcData::Dgram& dgram, PGPData* pgp_data) = 0;
protected:
    std::vector<XtcData::NamesId> m_namesId;
    XtcData::NamesLookup          m_namesLookup;
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

    T* create(const std::string& name, int nodeId)
    {
        auto it = m_create_funcs.find(name);
        if (it != m_create_funcs.end()) {
            return it->second(nodeId);
        }
        return nullptr;
    }

private:
    template <typename TDerived>
    static T* createFunc(int nodeId)
    {
        return new TDerived(nodeId);
    }

    typedef T* (*PCreateFunc)(int nodeId);
    std::unordered_map<std::string, PCreateFunc> m_create_funcs;
};

#endif // DETECTORS_H

#ifndef DETECTORS_H
#define DETECTORS_H

#include <string>
#include <unordered_map>
#include "xtcdata/xtc/Xtc.hh"

class Detector
{
public:
    virtual void configure(XtcData::Xtc& xtc) = 0;
    virtual void event(XtcData::Xtc& xtc, PGPData* pgp_data) = 0;
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
        return new TDerived();
    }

    typedef T* (*PCreateFunc)();
    std::unordered_map<std::string, PCreateFunc> m_create_funcs;
};

#endif // DETECTORS_H

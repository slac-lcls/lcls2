#ifndef DETECTORS_H
#define DETECTORS_H

#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <string>
#include <vector>
#include <unordered_map>

struct Parameters;

class Detector
{
public:
    Detector(Parameters* para) : m_para(para) {m_nodeId = m_para->tPrms.id;}
    virtual void connect() {};
    virtual unsigned configure(XtcData::Dgram& dgram) = 0;
    virtual void event(XtcData::Dgram& dgram, PGPData* pgp_data) = 0;
    XtcData::Dgram& transitionDgram() {return *(XtcData::Dgram*)m_dgrambuf;}
protected:
    Parameters* m_para;
    unsigned m_nodeId;
    std::vector<XtcData::NamesId> m_namesId;
    XtcData::NamesLookup m_namesLookup;
    uint8_t m_dgrambuf[XtcData::Dgram::MaxSize];
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

    T* create(Parameters* para)
    {
        std::string name = para->detectorType;
        auto it = m_create_funcs.find(name);
        if (it != m_create_funcs.end()) {
            return it->second(para);
        }
        return nullptr;
    }

private:
    template <typename TDerived>
    static T* createFunc(Parameters* para)
    {
        return new TDerived(para);
    }

    typedef T* (*PCreateFunc)(Parameters* para);
    std::unordered_map<std::string, PCreateFunc> m_create_funcs;
};

#endif // DETECTORS_H

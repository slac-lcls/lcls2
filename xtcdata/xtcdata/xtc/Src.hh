#ifndef XtcData_Src_hh
#define XtcData_Src_hh

#include "xtcdata/xtc/Level.hh"
#include <limits>
#include <stdint.h>

namespace XtcData
{

class Src
{
public:
    Src();
    Src(uint32_t value) : _value(value) {}
    Src(Level::Type level);

    uint32_t log() const;
    uint32_t phy() const;
    uint32_t nodeId() const {return _value;}

    Level::Type level() const;
    uint32_t    value() const;

    bool operator==(const Src& s) const;
    bool operator<(const Src& s) const;

    static uint32_t _sizeof();

    void phy(uint32_t value) { _value = value; }

  protected:
    uint32_t _log;   // cpo: eliminate this when we change xtc format
    uint32_t _value;
};

inline
XtcData::Src::Src() : _log(std::numeric_limits<uint32_t>::max()),
                      _phy(std::numeric_limits<uint32_t>::max())
{
}
inline
XtcData::Src::Src(XtcData::Level::Type level)
{
    uint32_t temp = (uint32_t)level;
    _log = (temp & 0xff) << 24;
}

inline
uint32_t XtcData::Src::log() const
{
    return _log;
}
inline
uint32_t XtcData::Src::phy() const
{
    return _phy;
}
inline
void XtcData::Src::phy(uint32_t value)
{
    _phy = value;
}
inline
XtcData::Level::Type XtcData::Src::level() const
{
    return (XtcData::Level::Type)((_log >> 24) & 0xff);
}
inline
uint32_t XtcData::Src::value() const
{
  return _log & ((1 << 24) - 1);
}

inline
bool XtcData::Src::operator==(const XtcData::Src& s) const
{
    return _phy == s._phy && _log == s._log;
}
inline
bool XtcData::Src::operator<(const XtcData::Src& s) const
{
    return (_phy < s._phy) || ((_phy == s._phy) && (_log < s._log));
}

inline
uint32_t XtcData::Src::_sizeof()
{
    return sizeof(XtcData::Src);
}
}
#endif

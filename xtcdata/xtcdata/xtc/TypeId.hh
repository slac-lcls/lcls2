#ifndef XtcData_TypeId_hh
#define XtcData_TypeId_hh

#include <stdint.h>

namespace XtcData
{

class TypeId
{
public:
    /*
     * Notice: New enum values should be appended to the end of the enum list, since
     *   the old values have already been recorded in the existing xtc files.
     */
    enum Type { Parent, ShapesData, Shapes, Data, Names, NumberOf };

    TypeId()
    {
    }
    TypeId(const TypeId& v);
    TypeId(Type type, unsigned version);
    TypeId(const char*);

    Type id() const;
    unsigned version() const;
    unsigned value() const;

    static const char* name(Type type);

private:
    enum { TypeBitMask    = 0x0fff };
    enum { VersionBitMask = 0xf000, VersionBitShift = 12 };
    uint16_t _value;
};


inline
XtcData::TypeId::TypeId(Type type, unsigned version)
    : _value(((version << VersionBitShift) & VersionBitMask) | type)
{
}

inline
XtcData::TypeId::TypeId(const TypeId& v) : _value(v._value)
{
}

inline
unsigned XtcData::TypeId::value() const
{
    return _value;
}

inline
unsigned XtcData::TypeId::version() const
{
    return (_value & VersionBitMask) >> VersionBitShift;
}

inline
XtcData::TypeId::Type XtcData::TypeId::id() const
{
    return (XtcData::TypeId::Type)(_value & TypeBitMask);
}

}

#endif

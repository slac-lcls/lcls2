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
    TypeId(Type type, uint32_t version);
    TypeId(const char*);

    Type id() const;
    uint32_t version() const;
    uint32_t value() const;

    static const char* name(Type type);
    static uint32_t _sizeof();

private:
    uint32_t _value;
};


inline
XtcData::TypeId::TypeId(Type type, uint32_t version)
  : _value((version << 16) | type)
{
}

inline
XtcData::TypeId::TypeId(const TypeId& v) : _value(v._value)
{
}

inline
uint32_t XtcData::TypeId::value() const
{
    return _value;
}

inline
uint32_t XtcData::TypeId::version() const
{
    return (_value & 0xffff0000) >> 16;
}

inline
XtcData::TypeId::Type XtcData::TypeId::id() const
{
    return (XtcData::TypeId::Type)(_value & 0xffff);
}

inline
uint32_t XtcData::TypeId::_sizeof()
{
    return sizeof(TypeId);
}
}

#endif

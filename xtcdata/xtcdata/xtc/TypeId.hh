/**
 * @file
 * @brief Declares XtcData::TypeId, the 16-bit type-and-version word stored in each Xtc header.
 */
#ifndef XtcData_TypeId_hh
#define XtcData_TypeId_hh

#include <stdint.h>

namespace XtcData
{

/** 16-bit word: bits 11:0 hold the Type, bits 15:12 the version. */
class TypeId
{
public:
    /*
     * Notice: New enum values should be appended to the end of the enum list, since
     *   the old values have already been recorded in the existing xtc files.
     */
    /** Payload types. Per the comment above, values are already recorded in xtc files, so new ones must be appended. */
    enum Type { Parent, /**< Value 0. NamesIter, ConfigIter and DataIter iterate into xtcs of this type. */ ShapesData, /**< Value 1. Set by the ShapesData constructor. */ Shapes, /**< Value 2. Set by the Shapes constructor. */ Data, /**< Value 3. Set by the Data constructor. */ Names, /**< Value 4. Set by the Names constructor; NamesIter reads xtcs of this type as Names. */ NumberOf  /**< Value 5, the number of types; name() returns "-Invalid-" for values at or above it. */ };

    /** Default constructor; the value is left uninitialized. */
    TypeId()
    {
    }
    /** Copy the 16-bit value of v. */
    TypeId(const TypeId& v);
    /** Set the value to ((version << 12) & 0xf000) | type, so only the low 4 bits of version are kept. */
    TypeId(Type type, unsigned version);
    /**
     * Parse a string of the form `NAME_vDIGITS`, where `NAME` is one of name()'s strings and `DIGITS` is the version number.
     * On a match the code computes (version << 16) | type, which the 16-bit storage truncates to just the type, so the parsed version is dropped (version() returns 0).
     * If there is no "_v" suffix, the digits are missing or followed by other characters, or the name is unknown, the value is NumberOf.
     */
    TypeId(const char*);

    /** Return bits 11:0 as a Type. */
    Type id() const;
    /** Return bits 15:12. */
    unsigned version() const;
    /** Return the full 16-bit value. */
    unsigned value() const;

    /** Return "Parent", "ShapesData", "Shapes", "Data" or "Names" for type, or "-Invalid-" if type >= NumberOf. */
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

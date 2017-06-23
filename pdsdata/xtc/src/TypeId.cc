#include "pdsdata/xtc/TypeId.hh"

#include <stdlib.h>
#include <string.h>

using namespace Pds;

TypeId::TypeId(Type type, uint32_t version, bool cmp)
: _value((version << 16) | type | (cmp ? 0x80000000 : 0))
{
}

TypeId::TypeId(const char* s) : _value(NumberOf)
{
    const char* token = strrchr(s, '_');
    if(!(token && *(token + 1) == 'v')) return;

    char* e;
    unsigned vsn = strtoul(token + 2, &e, 10);
    if(e == token + 2 || *e != 0) return;

    char* p = strndup(s, token - s);
    for(unsigned i = 0; i < NumberOf; i++)
        if(strcmp(p, name((Type)i)) == 0) _value = (vsn << 16) | i;
    free(p);
}

TypeId::TypeId(const TypeId& v) : _value(v._value)
{
}

uint32_t TypeId::value() const
{
    return _value;
}

uint32_t TypeId::version() const
{
    return (_value & 0xffff0000) >> 16;
}

TypeId::Type TypeId::id() const
{
    return (TypeId::Type)(_value & 0xffff);
}

bool TypeId::compressed() const
{
    return _value & 0x80000000;
}

const char* TypeId::name(Type type)
{
    static const char* _names[NumberOf] = { "Parent", "Data" };
    const char* p = (type < NumberOf ? _names[type] : "-Invalid-");
    if(!p) p = "-Unnamed-";
    return p;
}

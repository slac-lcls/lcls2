#include "xtcdata/xtc/TypeId.hh"

#include <stdlib.h>
#include <string.h>

using namespace XtcData;

TypeId::TypeId(const char* s) : _value(NumberOf)
{
    const char* token = strrchr(s, '_');
    if (!(token && *(token + 1) == 'v')) return;

    char* e;
    unsigned vsn = strtoul(token + 2, &e, 10);
    if (e == token + 2 || *e != 0) return;

    char* p = strndup(s, token - s);
    for (unsigned i = 0; i < NumberOf; i++)
        if (strcmp(p, name((Type)i)) == 0) _value = (vsn << 16) | i;
    free(p);
}

const char* TypeId::name(Type type)
{
    static const char* _names[NumberOf] = { "Parent", "ShapesData", "Shapes", "Data", "Names" };
    const char* p = (type < NumberOf ? _names[type] : "-Invalid-");
    if (!p) p = "-Unnamed-";
    return p;
}

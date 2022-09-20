#ifndef XtcData_Xtc_hh
#define XtcData_Xtc_hh

#include "Damage.hh"
#include "Src.hh"
#include "xtcdata/xtc/TypeId.hh"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>

#pragma pack(push,2)

namespace XtcData
{

class Xtc
{
public:
    Xtc() : damage(0), extent(0){};
    Xtc(const Xtc& xtc)
    : src(xtc.src), damage(xtc.damage), contains(xtc.contains), extent(sizeof(Xtc))
    {
    }
    Xtc(const TypeId& type) : damage(0), contains(type), extent(sizeof(Xtc))
    {
    }
    Xtc(const TypeId& type, const Src& _src)
    : src(_src), damage(0), contains(type), extent(sizeof(Xtc))
    {
    }
    Xtc(const TypeId& _tag, const Src& _src, unsigned _damage)
    : src(_src), damage(_damage), contains(_tag), extent(sizeof(Xtc))
    {
    }
    Xtc(const TypeId& _tag, const Src& _src, const Damage& _damage)
    : src(_src), damage(_damage), contains(_tag), extent(sizeof(Xtc))
    {
    }
    Xtc& operator=(const Xtc& xtc)
    {
      src      = xtc.src;
      damage   = xtc.damage;
      contains = xtc.contains;
      extent   = sizeof(Xtc);
      return *this;
    }
    void* operator new(size_t size)
    {
        return (void*)std::malloc(size);
    }
    void* operator new(size_t size, char* p, const void* end)
    {
      if (end && (&p[size] > end)) {
            printf("*** %s:%d: Insufficient space for %zu bytes\n",__FILE__,__LINE__,size);
            abort(); // Set gdb breakpoint to reported file:line to see how it got here
        }
        return (void*)p;
    }
    void* operator new(size_t size, Xtc* p, const void* end)
    {
        return p->alloc(size, end);
    }
    void* operator new(size_t size, Xtc& p, const void* end)
    {
        return p.alloc(size, end);
    }

    char* payload() const
    {
        return (char*)(this + 1);
    }
    int sizeofPayload() const
    {
        return extent - sizeof(Xtc);
    }
    Xtc* next()
    {
        return (Xtc*)((char*)this + extent);
    }
    const Xtc* next() const
    {
        return (const Xtc*)((char*)this + extent);
    }

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)

    void* alloc(size_t size, const void* end)
    {
        void* buffer = next();
        if (end && UNLIKELY((char*)buffer + size > end)){
            printf("*** %s:%d: Insufficient space for %zu bytes\n",__FILE__,__LINE__,size);
            abort(); // Set gdb breakpoint to reported file:line to see how it got here
        }
        extent += size;
        return buffer;
    }

#undef UNLIKELY

    Src      src;
    Damage   damage;
    TypeId   contains;
    uint32_t extent;
};
}

#pragma pack(pop)

#endif

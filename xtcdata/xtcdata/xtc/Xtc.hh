/**
 * @file
 * @brief Declares XtcData::Xtc, the 12-byte container header (src, damage, contains, extent) used at every level of an XTC2 datagram.
 */
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

/**
 * Container header, packed with 2-byte alignment: src (Src, 4 bytes), damage (Damage, 2 bytes), contains (TypeId, 2 bytes), extent (uint32_t).
 * extent is the size in bytes of this header plus its payload; the payload starts right after the header (payload()) and the next sibling starts extent bytes after the header (next()).
 */
class Xtc
{
/** Branch hint: expands to __builtin_expect(!!(expr), 0). Defined inside the class and undefined after alloc(). */
#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)

public:
    /** Set damage and extent to 0; src gets the Src default (value 0) and contains is left uninitialized. */
    Xtc() : damage(0), extent(0){};
    /** Copy src, damage and contains from xtc, but set extent to sizeof(Xtc) (empty payload). */
    Xtc(const Xtc& xtc)
    : src(xtc.src), damage(xtc.damage), contains(xtc.contains), extent(sizeof(Xtc))
    {
    }
    /** Header of the given type with default Src (value 0), damage 0 and extent sizeof(Xtc). */
    Xtc(const TypeId& type) : damage(0), contains(type), extent(sizeof(Xtc))
    {
    }
    /** Header with the given type and src, damage 0 and extent sizeof(Xtc). */
    Xtc(const TypeId& type, const Src& _src)
    : src(_src), damage(0), contains(type), extent(sizeof(Xtc))
    {
    }
    /** Header with the given type (_tag), src and damage word, and extent sizeof(Xtc). */
    Xtc(const TypeId& _tag, const Src& _src, unsigned _damage)
    : src(_src), damage(_damage), contains(_tag), extent(sizeof(Xtc))
    {
    }
    /** Header with the given type (_tag), src and damage, and extent sizeof(Xtc). */
    Xtc(const TypeId& _tag, const Src& _src, const Damage& _damage)
    : src(_src), damage(_damage), contains(_tag), extent(sizeof(Xtc))
    {
    }
    /**
     * Copy src, damage and contains from xtc and reset extent to sizeof(Xtc).
     * @return *this.
     */
    Xtc& operator=(const Xtc& xtc)
    {
      src      = xtc.src;
      damage   = xtc.damage;
      contains = xtc.contains;
      extent   = sizeof(Xtc);
      return *this;
    }
    /** Allocate size bytes with std::malloc. */
    void* operator new(size_t size)
    {
        return (void*)std::malloc(size);
    }
    /** Placement new at p. Aborts with a message if end is non-null and p + size would pass end. */
    void* operator new(size_t size, char* p, const void* end)
    {
        if (end && UNLIKELY((&p[size] > end))) {
            fprintf(stderr , "*** %s:%d: Insufficient space for %zu bytes (buffer %p, end %p)\n",
                    __FILE__,__LINE__,size,p,end);
            abort(); // Set gdb breakpoint to reported file:line to see how it got here
        }
        return (void*)p;
    }
    /** Placement new into the payload of p: returns p->alloc(size, end), which adds size to p's extent. */
    void* operator new(size_t size, Xtc* p, const void* end)
    {
        return p->alloc(size, end);
    }
    /** Placement new into the payload of p: returns p.alloc(size, end), which adds size to p's extent. */
    void* operator new(size_t size, Xtc& p, const void* end)
    {
        return p.alloc(size, end);
    }
    /** Free p with std::free. */
    void operator delete(void* p)       // Needed to avoid compiler warning
    {
      std::free(p);
    }
    /** Placement delete matching operator new(size_t, char*, const void*); does nothing. */
    void operator delete(void* ptr, char* p, const void* end) // Needed to avoid compiler warning
    {
      // Nothing to do
    }
    /** Placement delete matching operator new(size_t, Xtc*, const void*); does nothing. */
    void operator delete(void* ptr, Xtc* p, const void* end) // Needed to avoid compiler warning
    {
      // Nothing to do
    }
    /** Placement delete matching operator new(size_t, Xtc&, const void*); does nothing. */
    void operator delete(void* ptr, Xtc& p, const void* end) // Needed to avoid compiler warning
    {
      // Nothing to do
    }
    /** Return a pointer to the first byte after this header. */
    char* payload() const
    {
        return (char*)(this + 1);
    }
    //  This return type effectively limits extent to 31 bits
    /** Return extent - sizeof(Xtc) as an int (the comment above notes this limits extent to 31 bits). */
    int sizeofPayload() const
    {
        return extent - sizeof(Xtc);
    }
    /** Return the address extent bytes after the start of this header, where the next sibling Xtc begins. */
    Xtc* next()
    {
        return (Xtc*)((char*)this + extent);
    }
    /** Return the address extent bytes after the start of this header, where the next sibling Xtc begins (const version). */
    const Xtc* next() const
    {
        return (const Xtc*)((char*)this + extent);
    }
    /**
     * Reserve size bytes at next() by adding size to extent and return that address (memory is not initialized).
     * Aborts with a message if end is non-null and the reserved bytes would pass end.
     */
    void* alloc(size_t size, const void* end)
    {
        void* buffer = next();
        if (end && UNLIKELY((char*)buffer + size > end)){
            fprintf(stderr, "*** %s:%d: Insufficient space for %zu bytes (buffer %p, end %p, extent %u)\n",
                    __FILE__,__LINE__,size,buffer,end,extent);
            abort(); // Set gdb breakpoint to reported file:line to see how it got here
        }
        extent += size;
        return buffer;
    }

#undef UNLIKELY

    Src      src;  ///< Source word (Src); Names and ShapesData keep their NamesId here.
    Damage   damage;  ///< Damage word.
    TypeId   contains;  ///< TypeId (type and version) of the payload.
    uint32_t extent;  ///< Size in bytes of this header plus its payload.
};
}

#pragma pack(pop)

#endif

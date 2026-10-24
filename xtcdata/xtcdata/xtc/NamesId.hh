/**
 * @file
 * @brief Declares XtcData::NamesId, a Src value packing a 12-bit node id and an 8-bit names id.
 */
#ifndef XtcData_NamesId_hh
#define XtcData_NamesId_hh

#include "xtcdata/xtc/Src.hh"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

namespace XtcData
{

// this class translates a nodeId and a namesId into a dense number
// (stored in the Src of Names and ShapesData).  This is number is
// used to do an array lookup to find the correct Names object associated
// with a ShapesData object.  The Names object is in the configure
// transition, while the ShapesData object is in the event.  One could
// also use a map instead of an array to do the lookup.  That would save
// memory, but be slower.

/**
 * Src whose value holds a node id in bits 19:8 and a names id in bits 7:0 (level Segment).
 * Per the comment above, Names and ShapesData store it in their Src so a ShapesData can be matched to its Names.
 */
class NamesId: public Src
{
public:
    // must be consistent with the total number of bits used below.
    // this will determine the size of the Names lookup array,
    // so we try not to make it too large
    /** Size of the NamesId value space. The comment above says it must match the bits used and sets the size of the Names lookup. */
    enum {NumberOf=1<<20 /**< 1<<20, the number of distinct 12-bit node id + 8-bit names id values. */ };

    /** Default constructor: Src() gives value 0, level Segment (the source comment calls it an undefined Src). */
    NamesId() : Src() {}  // undefined Src

    /**
     * Store ((nodeId & 0xfff) << 8) | (namesId & 0xff) as the Src value, level Segment.
     * Prints a message and calls abort() if nodeId does not fit in 12 bits or namesId in 8 bits.
     */
    NamesId(unsigned nodeId, unsigned namesId) : Src(((nodeId&0xfff)<<8)|(namesId&0xff)) {
        if ((nodeId&0xfff) != nodeId) {
            printf("*** %s:%d: nodeId too large %d\n",__FILE__,__LINE__,nodeId);
            abort();
        }
        if ((namesId&0xff) != namesId) {
            printf("*** %s:%d: namesId too large %d\n",__FILE__,__LINE__,namesId);
            abort();
        }
    }

    /** Return value(), the packed node id and names id. */
    operator unsigned() const {return value();}

    /** Return bits 19:8. */
    unsigned nodeId()  {return (_value>>8)&0xfff;}
    /** Return bits 7:0. */
    unsigned namesId() {return _value&0xff;}

};

}
#endif

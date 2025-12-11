#ifndef XtcData_NamesId_hh
#define XtcData_NamesId_hh

#include "Src.hh"
#include <stdint.h>

namespace XtcData
{

// this class translates a nodeId and a namesId into a dense number
// (stored in the Src of Names and ShapesData).  This is number is
// used to do an array lookup to find the correct Names object associated
// with a ShapesData object.  The Names object is in the configure
// transition, while the ShapesData object is in the event.  One could
// also use a map instead of an array to do the lookup.  That would save
// memory, but be slower.

class NamesId: public Src
{
public:
    // must be consistent with the total number of bits used below.
    // this will determine the size of the Names lookup array,
    // so we try not to make it too large
    enum {NumberOf=1<<20};

    NamesId() : Src() {}  // undefined Src

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

    operator unsigned() const {return value();}

    unsigned nodeId()  {return (_value>>8)&0xfff;}
    unsigned namesId() {return _value&0xff;}

};

}
#endif

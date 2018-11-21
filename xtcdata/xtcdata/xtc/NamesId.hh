#ifndef XtcData_NamesId_hh
#define XtcData_NamesId_hh

#include "xtcdata/xtc/Src.hh"
#include <stdint.h>

namespace XtcData
{

// this class translates a nodeId and a namesId into a dense number
// (stored in the Src of Names and ShapesData).  This is number is
// used to do an array lookup to find the correct Names object associated
// with a ShapesData object.

class NamesId: public Src
{
public:
    enum {NumberOf=1<<12}; // must be consistent with masks below

    NamesId(unsigned nodeId, unsigned namesId) : Src((nodeId&0xffff)<<8|(namesId&0xff)) {
        assert ((nodeId&0xffff) == nodeId);
        assert ((namesId&0xff)  == namesId);
    }

    unsigned nodeId()  {return (_value>>8)&0xffff;}
    unsigned namesId() {return _value&0xff;}
    unsigned value()   {return _value;}

};

}
#endif

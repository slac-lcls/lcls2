#ifndef XtcData_NamesVec_hh
#define XtcData_NamesVec_hh

#include "xtcdata/xtc/NameIndex.hh"
#include "xtcdata/xtc/NamesId.hh"

#include <vector>

namespace XtcData{

// This class is fundamental to self-describing xtc data.  It is used
// to associate the Names xtc on the configure transition with the
// ShapesData xtc that shows up every event. The Names and ShapesData
// xtc's each get a unique identifier that can be used to associate
// the two (the NamesId class which is put in the xtc Src field).
// This identifier is used as an index into the fixed-length NamesVec
// array below.  A map would have been more memory-efficient, but the
// vector provides a faster lookup, which is preferable since this is
// is done multiple times per event.

class NamesVec
{
public:
    NamesVec() : _namesVec(NamesId::NumberOf) {}
    NameIndex& operator[] (NamesId namesId) {
        assert (namesId.value()<_namesVec.size());
        return _namesVec[namesId.value()];
    }
    NameIndex& operator[] (unsigned i) {
        assert (i<_namesVec.size());
        return _namesVec[i];
    }
    unsigned size() {return _namesVec.size();}
    
private:
    std::vector<NameIndex> _namesVec;
};

};

# endif

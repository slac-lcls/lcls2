#ifndef XtcData_NamesVec_hh
#define XtcData_NamesVec_hh

#include "xtcdata/xtc/NameIndex.hh"
#include "xtcdata/xtc/NamesId.hh"

#include <vector>

namespace XtcData{

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

#include "xtcdata/xtc/NamesIter.hh"

using namespace XtcData;

int NamesIter::process(Xtc* xtc)
{
    switch (xtc->contains.id()) {
    case (TypeId::Parent): {
        iterate(xtc); // look inside anything that is a Parent
        break;
    }
    case (TypeId::Names): {
        Names& names = *(Names*)xtc;
        NamesId& namesId = names.namesId();
        if (_namesLookup.find(namesId) != _namesLookup.end()) {
            printf("NamesIter.cc: Found duplicate namesId 0x%x\n",namesId);
            throw "NamesIter.cc: Found duplicate namesId 0x%x";
        }
        _namesLookup[namesId] = NameIndex(names);
        break;
    }
    default:
        break;
    }
    return Continue;
}

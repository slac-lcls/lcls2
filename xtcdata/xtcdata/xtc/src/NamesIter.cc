#include "xtcdata/xtc/NamesIter.hh"

using namespace XtcData;

int NamesIter::process(Xtc* xtc)
{
    // printf("found typeid %s\n",XtcData::TypeId::name(xtc->contains.id()));
    switch (xtc->contains.id()) {
    case (TypeId::Parent): {
        iterate(xtc); // look inside anything that is a Parent
        break;
    }
    case (TypeId::Names): {
        Names& names = *(Names*)xtc;
        NamesId& namesId = names.namesId();
        _namesVec[namesId] = NameIndex(names);
        break;
    }
    default:
        break;
    }
    return Continue;
}

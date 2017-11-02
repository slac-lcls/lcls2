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
        _namesVec.push_back(NameIndex(*(Names*)xtc));
        break;
    }
    default:
        break;
    }
    return Continue;
}

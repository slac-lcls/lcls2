#include "xtcdata/xtc/Smd.hh"

#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/XtcIterator.hh"

#include <iostream>
#include <stdlib.h>


using namespace XtcData;
using namespace std;

class SmdDef:public VarDef
{
public:
  enum index
    {
      intOffset,
      intDgramSize
    };

   SmdDef()
   {
     NameVec.push_back({"intOffset", Name::UINT64});
     NameVec.push_back({"intDgramSize", Name::UINT64});
   }
} SmdDef;

class CheckNamesIdIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    CheckNamesIdIter(NamesId offset_namesId) : XtcIterator(),
                                               _offset_namesId(offset_namesId) {}

    int process(Xtc* xtc, const void* bufEnd)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc, bufEnd);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            if (names.namesId() == _offset_namesId) {
                printf ("Error: smd names id in use\n");
                throw ("smd names id in use");
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    NamesId _offset_namesId;
};

void addNames(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId namesId)
{
    Alg alg("offsetAlg",0,0,0);

    // check that our chose namesId isn't already in use
    CheckNamesIdIter checkNamesId(namesId);
    checkNamesId.iterate(&parent, bufEnd);

    Names& offsetNames = *new(parent, bufEnd) Names(bufEnd, "smdinfo", alg, "offset", "", namesId);
    offsetNames.add(parent,bufEnd,SmdDef);
    namesLookup[namesId] = NameIndex(offsetNames);
}

Dgram* Smd::generate(Dgram* dgIn, void* buf, const void* bufEnd, uint64_t offset, uint64_t size,
        NamesLookup& namesLookup, NamesId namesId)
{
    if (dgIn->service() != TransitionId::L1Accept) {
        Dgram *dgOut;
        dgOut = (Dgram*)buf;

        if (dgIn->service() == TransitionId::EndRun) {
            //  EndRun has the cube results.  Don't want any EndRun payload in smd
            new (buf) Dgram(*dgIn, dgIn->xtc);
        }
        else {
            memcpy(dgOut, dgIn, sizeof(*dgIn));
            memcpy(dgOut->xtc.payload(), dgIn->xtc.payload(), dgIn->xtc.sizeofPayload());
            if (dgIn->service() == TransitionId::Configure) {
                addNames(dgOut->xtc, bufEnd, namesLookup, namesId);
            }
        }
        return dgOut;

    } else {
        Dgram& dgOut = *(Dgram*)buf;
        dgOut.time = dgIn->time;
        dgOut.env = dgIn->env;
        dgOut.xtc = {{TypeId::Parent, 0}};

        CreateData createSmd(dgOut.xtc, bufEnd, namesLookup, namesId);
        createSmd.set_value(SmdDef::intOffset, offset);
        createSmd.set_value(SmdDef::intDgramSize, size);

        if (offset < 0) {
            cout << "Error offset value (offset=" << offset << ")" << endl;
        }
        if (size <= 0) {
            cout << "Error size value (size=" << size << ")" << endl;
        }

        return &dgOut;

    } // end else dgIn->service()

}

#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"

namespace XtcData{
class NamesIter : public XtcData::XtcIterator
{
public:
    enum { Stop, Continue };
    NamesIter(XtcData::Xtc* xtc, const void* bufEnd) : XtcData::XtcIterator(xtc, bufEnd) {}
    NamesIter() : XtcData::XtcIterator() {}
    virtual int process(XtcData::Xtc* xtc, const void* bufEnd);
    NamesLookup& namesLookup() {return _namesLookup;}
private:
    NamesLookup _namesLookup;
};
};

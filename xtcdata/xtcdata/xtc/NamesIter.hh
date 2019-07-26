#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"

namespace XtcData{
class NamesIter : public XtcData::XtcIterator
{
public:
    enum { Stop, Continue };
    NamesIter(XtcData::Xtc* xtc) : XtcData::XtcIterator(xtc) {}
    NamesIter() : XtcData::XtcIterator() {}
    virtual int process(XtcData::Xtc* xtc);
    NamesLookup& namesLookup() {return _namesLookup;}
private:
    NamesLookup _namesLookup;
};
};

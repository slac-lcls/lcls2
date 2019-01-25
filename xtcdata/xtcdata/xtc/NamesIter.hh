#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesVec.hh"

namespace XtcData{
class NamesIter : public XtcData::XtcIterator
{
public:
    enum { Stop, Continue };
    NamesIter(XtcData::Xtc* xtc) : XtcData::XtcIterator(xtc) {}
    NamesIter() : XtcData::XtcIterator() {}
    int process(XtcData::Xtc* xtc);
    NamesVec& namesVec() {return _namesVec;}
private:
    NamesVec _namesVec;
};
};

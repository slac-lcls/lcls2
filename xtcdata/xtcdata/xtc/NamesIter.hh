#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/DescData.hh"

namespace XtcData{
class NamesIter : public XtcData::XtcIterator
{
public:
    enum { Stop, Continue };
    NamesIter(XtcData::Xtc* xtc) : XtcData::XtcIterator(xtc) {}
    NamesIter() : XtcData::XtcIterator() {}
    int process(XtcData::Xtc* xtc);
    std::vector<NameIndex>& namesVec() {return _namesVec;}
private:
    std::vector<NameIndex> _namesVec;
};
};

/*
 * class ConfigIter provides acess to configuration info from the 1st datagram in xtc2 file.
 *
 */

#include "xtcdata/xtc/NamesIter.hh"
//#include "xtcdata/xtc/DescData.hh"
//#include "xtcdata/xtc/NamesLookup.hh"

namespace XtcData{
class ConfigIter : public XtcData::NamesIter
{
public:
  //ConfigIter(XtcData::Xtc* xtc) : XtcData::NamesIter(xtc), _desc_shape(NULL), _desc_value(NULL) { iterate(); }
    ConfigIter(XtcData::Xtc* xtc) : XtcData::NamesIter(xtc) { iterate(); }
    ConfigIter() : XtcData::NamesIter() {}
   ~ConfigIter();

    int process(XtcData::Xtc* xtc);

    ShapesData& shape() {return *_shapesData[0];}
    ShapesData& value() {return *_shapesData[1];}
    //NamesLookup& namesLookup() {return _namesLookup;} // defined in super-class NamesIter
    //void iterate();                                   // defined in super-super-class XtcIterator

    DescData& desc_shape();
    DescData& desc_value();

    ConfigIter(const ConfigIter&) = delete;
    ConfigIter& operator = (const ConfigIter&) = delete;

private:
    ShapesData* _shapesData[2];
    DescData* _desc_shape = NULL;
    DescData* _desc_value = NULL;
};
};

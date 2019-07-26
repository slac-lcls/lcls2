#ifndef XTCDATA_CONFIGITER_H
#define XTCDATA_CONFIGITER_H

/*
 * class ConfigIter provides acess to configuration info from the 1st datagram in xtc2 file.
 *
 */

#include "xtcdata/xtc/NamesIter.hh"
//#include "xtcdata/xtc/DescData.hh"
//#include "xtcdata/xtc/NamesLookup.hh"

namespace XtcData{

//class XtcData::NamesIter;

class ConfigIter : public XtcData::NamesIter
{
public:

    enum CTOR_TYPE {CTOR_DEFAULT, CTOR_REGULAR};

    ConfigIter(XtcData::Xtc* xtc) : XtcData::NamesIter(xtc), _ctor_type(CTOR_REGULAR) {iterate();}
    ConfigIter() : XtcData::NamesIter(), _ctor_type(CTOR_DEFAULT) {}
   ~ConfigIter();

    virtual int process(XtcData::Xtc* xtc);

    ShapesData& shape() {return *_shapesData[0];}
    ShapesData& value() {return *_shapesData[1];}
    //NamesLookup& namesLookup() {return _namesLookup;} // defined in super-class NamesIter
    //void iterate();                                   // defined in super-super-class XtcIterator

    DescData& desc_shape();
    DescData& desc_value();

    ConfigIter(const ConfigIter&) = delete;
    ConfigIter& operator = (const ConfigIter&) = delete;

    bool default_constructor() const {return _ctor_type==CTOR_DEFAULT;}
    bool regular_constructor() const {return _ctor_type==CTOR_REGULAR;}
    CTOR_TYPE constructor_type() const {return _ctor_type;}

private:
    ShapesData* _shapesData[2];
    DescData* _desc_shape = NULL;
    DescData* _desc_value = NULL;
    CTOR_TYPE _ctor_type;
};
};

#endif // 

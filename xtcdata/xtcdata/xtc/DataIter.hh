#ifndef XTCDATA_DATAITER_H
#define XTCDATA_DATAITER_H

/*
 * class DataIter provides acess to data in datagrams of xtc2 file.
 *
 */

#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"

namespace XtcData{

class DataIter : public XtcData::XtcIterator
{
public:
    enum {Stop, Continue};

    DataIter(XtcData::Xtc* xtc) : XtcData::XtcIterator(xtc) { iterate(); }
    DataIter() : XtcData::XtcIterator() {}
   ~DataIter();

    virtual int process(XtcData::Xtc* xtc);

    ShapesData& shape() {return *_shapesData[0];}
    ShapesData& value() {return *_shapesData[1];}
    //void iterate(); // defined in super-class XtcIterator

    DescData& desc_shape(NamesLookup& names_map);
    DescData& desc_value(NamesLookup& names_map);

    DataIter(const DataIter&) = delete;
    DataIter& operator = (const DataIter&) = delete;

private:
    ShapesData* _shapesData[2];
    DescData* _desc_shape = NULL;
    DescData* _desc_value = NULL;
};
};

#endif // 


#include "xtcdata/xtc/ConfigIter.hh"

using namespace XtcData;

//ConfigIter::ConfigIter(XtcData::Xtc* xtc) : XtcData::NamesIter(xtc) { iterate(); }
//ConfigIter::ConfigIter() : XtcData::NamesIter() {}

ConfigIter::~ConfigIter() { 
    if(_desc_shape) delete _desc_shape; 
    if(_desc_value) delete _desc_value;
    _desc_shape = NULL;
    _desc_value = NULL;
}

int ConfigIter::process(XtcData::Xtc* xtc)
{
  TypeId::Type type = xtc->contains.id(); 
  //printf("DataIter TypeId::%-20s Xtc*: %p\n", TypeId::name(type), &xtc);

  switch (type) {
    case (TypeId::Parent): {
        iterate(xtc);
        break;
    }
    case (TypeId::Names): {
        Names& names = *(Names*)xtc;
        NamesId& namesId = names.namesId();
        NamesLookup& _namesLookup = namesLookup();
        _namesLookup[namesId] = NameIndex(names);
        break;
    }
    case (TypeId::ShapesData): {
	ShapesData* psd = (ShapesData*)xtc;
        NamesId namesId = psd->namesId();
        _shapesData[namesId.namesId()] = psd;
        break;
    }
    default:
	break;
  }
  return Continue;
}

DescData& ConfigIter::desc_shape() {
  if(! _desc_shape) _desc_shape = new DescData(shape(), namesLookup()[shape().namesId()]);
  return *_desc_shape; 
}

DescData& ConfigIter::desc_value() {
  if(! _desc_value) _desc_value = new DescData(value(), namesLookup()[value().namesId()]);
  return *_desc_value; 
}

//---------

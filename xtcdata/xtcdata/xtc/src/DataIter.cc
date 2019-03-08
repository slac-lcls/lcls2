
#include "xtcdata/xtc/DataIter.hh"

using namespace XtcData;

DataIter::~DataIter() { 
    if(_desc_shape) delete _desc_shape; 
    if(_desc_value) delete _desc_value;
    _desc_shape = NULL;
    _desc_value = NULL;
}

int DataIter::process(Xtc* xtc)
{
  TypeId::Type type = xtc->contains.id(); 
  //printf("DataIter TypeId::%-20s Xtc*: %p\n", TypeId::name(type), &xtc);

  switch (type) {
    case (TypeId::Parent): {
	iterate(xtc); 
        break;
    }
    case (TypeId::Names): {
        break;
    }
    case (TypeId::ShapesData): {
	ShapesData* psd = (ShapesData*)xtc;
        NamesId namesId = psd->namesId();
        _shapesData[namesId.namesId()] = psd;
        break;
    }
      //case (TypeId::Shapes): {break;}
      //case (TypeId::Data):   {break;}
    default:
       break;
  }
  return Continue;
}

DescData& DataIter::desc_shape(NamesLookup& names_map) {
  if(! _desc_shape) _desc_shape = new DescData(shape(), names_map[shape().namesId()]);
  return *_desc_shape; 
}

DescData& DataIter::desc_value(NamesLookup& names_map) {
  if(! _desc_value) _desc_value = new DescData(value(), names_map[value().namesId()]);
  return *_desc_value; 
}

//---------

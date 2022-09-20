

#include <stdio.h> // sprintf, printf( "%lf\n", accum );

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

int ConfigIter::process(XtcData::Xtc* xtc, const void* bufEnd)
{
  TypeId::Type type = xtc->contains.id();
  //printf("\nZZZ ConfigIter.process TypeId::%-20s Xtc*: %p", TypeId::name(type), &xtc);

  switch (type) {
    case (TypeId::Parent): {
        iterate(xtc, bufEnd);
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
  //printf("\n YYY ConfigIter::desc_shape namesLookup(): %d", namesLookup().size());
  //printf("\n YYY ConfigIter::desc_shape: %p", _desc_shape);
  //if(! _desc_shape) _desc_shape = new DescData(shape(), namesLookup()[shape().namesId()]);
  if(_desc_shape) delete _desc_shape;
  _desc_shape = new DescData(shape(), namesLookup()[shape().namesId()]);
  return *_desc_shape;
}

//void ConfigIter::desc_value() {
DescData& ConfigIter::desc_value() {
  printf("\nYYY ==> ConfigIter::desc_value");

  // problem arises here, value() returns undefined _shapesData[1]
  //ShapesData& v = value();
  //printf("\nYYY == ConfigIter::desc_value - value is accessed");
  //printf("\nYYY == ConfigIter::desc_value - value().namesId(): %i", v.namesId());

  //if(_desc_value){delete _desc_value; _desc_value = NULL;}
  //printf("\nYYY ConfigIter::desc_value: %p", _desc_value);
  //printf("\n YYY ConfigIter::desc_value namesLookup(): %d", namesLookup().size());

  if(_desc_value) delete _desc_value;
  _desc_value = new DescData(value(), namesLookup()[value().namesId()]);
  return *_desc_value;
}

//---------

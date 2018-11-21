#ifndef XtcData_NameIndex_hh
#define XtcData_NameIndex_hh

#include "xtcdata/xtc/ShapesData.hh"

#include <map>

typedef std::map<std::string, unsigned> IndexMap;

namespace XtcData
{

class NameIndex {
public:
    // placeholder instance. enables fast array lookups to find a NameIndex.
    NameIndex() : _names(0) {}

    NameIndex(Names& names) {
        _init_names(names); 
        unsigned iarray = 0;
        for (unsigned i=0; i<_names->num(); i++) {
            Name& name = _names->get(i);
            _nameMap[std::string(name.name())]=i;
            if (name.rank()>0) {
                _shapeMap[std::string(name.name())]=iarray;
                iarray++;
            }
        }
    }
    NameIndex(const NameIndex& old) {
        if (old._names) {
            _init_names(*old._names);
        } else {
            _names = old._names;
        }
        _shapeMap = old._shapeMap;
        _nameMap = old._nameMap;
    }
    NameIndex& operator=(const NameIndex& rhs) {
        if (_names) free(_names);
        if (rhs._names) {
            _names = (Names*)malloc(rhs._names->extent);
            std::memcpy(_names, rhs._names, rhs._names->extent);
        } else {
            _names = rhs._names; // copy over the zero
        }
        _shapeMap = rhs._shapeMap;
        _nameMap = rhs._nameMap;
    }
    ~NameIndex() {if (_names) free(_names);}
    IndexMap& shapeMap() {return _shapeMap;}
    IndexMap& nameMap()  {return _nameMap;}
    Names&    names()    {return *_names;}
    bool      exists()   {return _names!=0;}
private:
    void _init_names(Names& names) {
        _names = (Names*)malloc(names.extent);
        std::memcpy(_names, &names, names.extent);
    }
    Names*   _names;
    IndexMap _shapeMap;
    IndexMap _nameMap;
};

}

#endif

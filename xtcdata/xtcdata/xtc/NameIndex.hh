#ifndef XtcData_NameIndex_hh
#define XtcData_NameIndex_hh

#include "xtcdata/xtc/ShapesData.hh"

#include <map>

typedef std::map<std::string, unsigned> IndexMap;

namespace XtcData
{

class NameIndex {
public:
    // default constructor, used by NamesLookup std::map for keys
    // that don't exist (see comment in names() method below).
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
            std::memcpy((void*)_names, (const void*)rhs._names, rhs._names->extent);
        } else {
            _names = rhs._names; // copy over the zero
        }
        _shapeMap = rhs._shapeMap;
        _nameMap = rhs._nameMap;
        return *this;
    }
    ~NameIndex() {if (_names) free(_names);}
    IndexMap& shapeMap() {return _shapeMap;}
    IndexMap& nameMap()  {return _nameMap;}
    Names&    names()    {
        if (_names == 0) {
            // this typically happens when the user gives a bad NamesId
            // to NamesLookup[NamesId], which is in turn often caused by the
            // NamesLookup being empty (NamesLookup should be filled in on the
            // configure transition).  The reason the _names pointer is
            // zero is that std::map (implementation of NamesLookup) uses
            // the default NameIndex constructor for keys that don't exist.
            // We considered throwing an error in the default constructor,
            // but that constructor is also used for assignments like
            // NamesLookups[NamesId] = <...> which shouldn't throw.  That
            // syntax could be replaced by std::map.insert() to use
            // copy constructors, but the syntax is significantly messier.
            //
            // Ideally this error would be at a higher level in the code,
            // but this was the earliest common place we could
            // find to put it that didn't make interfaces more complex.
            //
            // - cpo and sioan
            printf("NameIndex.hh: _names is 0.  Typically caused by bad "
                   "NamesId given to NamesLookup[NamesId], or empty NamesLookup\n");
            throw "NameIndex.hh: _names is 0. ";
        }
        return *_names;
    }
    bool      exists()   {return _names!=0;}
private:
    void _init_names(Names& names) {
        _names = (Names*)malloc(names.extent);
        std::memcpy((void*)_names, (const void*)&names, names.extent);
    }
    Names*   _names;
    IndexMap _shapeMap;
    IndexMap _nameMap;
};

}

#endif

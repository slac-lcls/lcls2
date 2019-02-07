#ifndef XtcData_NamesLookup_hh
#define XtcData_NamesLookup_hh

#include "xtcdata/xtc/NameIndex.hh"
#include "xtcdata/xtc/NamesId.hh"

#include <map>

namespace XtcData{

// This class is fundamental to self-describing xtc data.  It is used
// to associate the Names xtc on the configure transition with the
// ShapesData xtc that shows up every event. The Names and ShapesData
// xtc's each get a unique identifier that can be used to associate
// the two (the NamesId class which is put in the xtc Src field).
// This identifier is used as a key in the map below.  An earlier
// implementation was done with std::vector, to avoid the map key
// search, but used alot of memory (see NamesId::NumberOf)

typedef std::map<unsigned,NameIndex> NamesLookup;

};

# endif

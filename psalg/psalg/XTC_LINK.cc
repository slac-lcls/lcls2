#include "xtcdata/xtc/Array.hh"
// #include "xtcdata/xtc/ConfigIter.hh"    TODO: Pinpoint the redefinition issue between this and NamesIter.hh. We’re hitting a class redefinition error because the compiler sees the same header (NamesIter.hh) from two different include paths (source and install), causing it to treat them as distinct files.
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/DataIter.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Level.hh"
// #include "xtcdata/xtc/NamesIter.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/Src.hh"
#include "xtcdata/xtc/TransitionId.hh"
#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"

int main() {
}
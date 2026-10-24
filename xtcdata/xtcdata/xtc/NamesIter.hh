/**
 * @file
 * @brief Declares XtcData::NamesIter, an XtcIterator that collects Names xtcs into a NamesLookup.
 */
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"

/** Namespace of the XTC data-format classes (Xtc, Dgram, TypeId, Src, NamesId, ShapesData, DescData, iterators, ...). */
namespace XtcData{
/** XtcIterator subclass that records each Names xtc it visits into a NamesLookup, descending into Parent xtcs. */
class NamesIter : public XtcData::XtcIterator
{
public:
    /** Return values of process(). XtcIterator::iterate() stops when process() returns 0 (Stop). */
    enum { Stop, /**< Value 0: stop iterating. */ Continue  /**< Value 1: keep iterating. */ };
    /** Construct an iterator rooted at xtc; bufEnd is passed to XtcIterator as the end-of-buffer bound (may be null). */
    NamesIter(XtcData::Xtc* xtc, const void* bufEnd) : XtcData::XtcIterator(xtc, bufEnd) {}
    /** Default constructor; the XtcIterator root is not set, so use iterate(Xtc*, const void*). */
    NamesIter() : XtcData::XtcIterator() {}
    /**
     * Handle one child xtc: iterate into TypeId::Parent xtcs and store a NameIndex built from each TypeId::Names xtc under its NamesId; ignore other types.
     * If a NamesId is already present it prints a message and throws a const char* exception.
     * @return Always Continue.
     */
    virtual int process(XtcData::Xtc* xtc, const void* bufEnd);
    /** Return a reference to the NamesLookup filled by process(). */
    NamesLookup& namesLookup() {return _namesLookup;}
private:
    NamesLookup _namesLookup;
};
};

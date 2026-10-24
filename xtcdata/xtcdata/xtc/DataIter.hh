/**
 * @file
 * @brief Declares XtcData::DataIter, an XtcIterator that records ShapesData xtcs of a datagram.
 */
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

/**
 * XtcIterator subclass that stores pointers to ShapesData xtcs in a 2-entry array indexed by NamesId::namesId() (low 8 bits of the NamesId).
 * The header comment says it gives access to data in datagrams of an xtc2 file.
 */
class DataIter : public XtcData::XtcIterator
{
public:
    /** Return values of process(). XtcIterator::iterate() stops when process() returns 0 (Stop). */
    enum {Stop, /**< Value 0: stop iterating. */ Continue /**< Value 1: keep iterating. */ };

    /** Construct rooted at xtc and immediately call iterate(); bufEnd bounds the buffer (may be null). */
    DataIter(XtcData::Xtc* xtc, const void* bufEnd) : XtcData::XtcIterator(xtc, bufEnd) { iterate(); }
    /** Default constructor; nothing is iterated and the ShapesData array is left uninitialized. */
    DataIter() : XtcData::XtcIterator() {}
   /** Delete the DescData objects created by desc_shape() and desc_value(). */
   ~DataIter();

    /**
     * Iterate into TypeId::Parent xtcs and store each TypeId::ShapesData pointer at index NamesId::namesId(); other types (including Names) are ignored.
     * There is no bounds check, so a namesId of 2 or more writes past the 2-entry array.
     * @return Always Continue.
     */
    virtual int process(XtcData::Xtc* xtc, const void* bufEnd);

    /** Return the ShapesData stored at index 0 (not checked to have been set). */
    ShapesData& shape() {return *_shapesData[0];}
    /** Return the ShapesData stored at index 1 (not checked to have been set). */
    ShapesData& value() {return *_shapesData[1];}
    //void iterate(); // defined in super-class XtcIterator

    /**
     * On the first call build a DescData from shape() and names_map[shape().namesId()]; later calls return the same cached object and ignore names_map.
     * The object is owned by this DataIter and freed by the destructor.
     */
    DescData& desc_shape(NamesLookup& names_map);
    /**
     * On the first call build a DescData from value() and names_map[value().namesId()]; later calls return the same cached object and ignore names_map.
     * The object is owned by this DataIter and freed by the destructor.
     */
    DescData& desc_value(NamesLookup& names_map);

    /** Copy construction is disabled. */
    DataIter(const DataIter&) = delete;
    /** Copy assignment is disabled. */
    DataIter& operator = (const DataIter&) = delete;

private:
    ShapesData* _shapesData[2];
    DescData* _desc_shape = NULL;
    DescData* _desc_value = NULL;
};
};

#endif //

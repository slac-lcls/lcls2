/**
 * @file
 * @brief Declares XtcData::ConfigIter, a NamesIter that also records ShapesData xtcs.
 */
#ifndef XTCDATA_CONFIGITER_H
#define XTCDATA_CONFIGITER_H

/*
 * class ConfigIter provides acess to configuration info from the 1st datagram in xtc2 file.
 *
 */

#include "xtcdata/xtc/NamesIter.hh"
#include <unordered_map>
//#include "xtcdata/xtc/DescData.hh"
//#include "xtcdata/xtc/NamesLookup.hh"

namespace XtcData{

//class XtcData::NamesIter;

/**
 * NamesIter subclass that, besides Names, keeps pointers to ShapesData xtcs keyed by the low 8 bits (NamesId::namesId()) of their NamesId.
 * The header comment says it gives access to configuration info from the first datagram of an xtc2 file.
 */
class ConfigIter : public XtcData::NamesIter
{
public:

    /** Records which constructor built the object. */
    enum CTOR_TYPE {CTOR_DEFAULT, /**< Set by ConfigIter(). */ CTOR_REGULAR /**< Set by ConfigIter(Xtc*, const void*). */ };

    /** Construct rooted at xtc and immediately call iterate(); bufEnd bounds the buffer (may be null). */
    ConfigIter(XtcData::Xtc* xtc, const void* bufEnd) : XtcData::NamesIter(xtc, bufEnd), _ctor_type(CTOR_REGULAR) {iterate();}
    /** Default constructor; nothing is iterated and constructor_type() returns CTOR_DEFAULT. */
    ConfigIter() : XtcData::NamesIter(), _ctor_type(CTOR_DEFAULT) {}
   /** Delete the DescData objects created by desc_shape() and desc_value(). */
   ~ConfigIter();

    /**
     * Iterate into TypeId::Parent xtcs, store a NameIndex for each TypeId::Names xtc in namesLookup() (a duplicate NamesId overwrites, no error), and store each TypeId::ShapesData pointer under NamesId::namesId().
     * @return Always Continue.
     */
    virtual int process(XtcData::Xtc* xtc, const void* bufEnd);

    /** Return the ShapesData stored under namesId 0. No check is made: a missing entry yields a null pointer that is dereferenced. */
    ShapesData& shape() {return *_shapesData[0];}
    /** Return the ShapesData stored under namesId 1. No check is made: a missing entry yields a null pointer that is dereferenced. */
    ShapesData& value() {return *_shapesData[1];}
    /** Return the ShapesData stored under namesId idx. No check is made: a missing entry yields a null pointer that is dereferenced. */
    ShapesData& getShape(unsigned idx) {return *_shapesData[idx];}
    //NamesLookup& namesLookup() {return _namesLookup;} // defined in super-class NamesIter
    //void iterate();                                   // defined in super-super-class XtcIterator

    /**
     * Delete any previous result, then build and return a new DescData from shape() and the NameIndex that namesLookup() holds for its NamesId.
     * The object is owned by this ConfigIter and freed by the next call or the destructor.
     */
    DescData& desc_shape();
    /**
     * Print a debug line ("YYY ==> ConfigIter::desc_value") to stdout, delete any previous result, then build and return a new DescData from value() and its NameIndex in namesLookup().
     * The object is owned by this ConfigIter and freed by the next call or the destructor.
     */
    DescData& desc_value();

    /** Copy construction is disabled. */
    ConfigIter(const ConfigIter&) = delete;
    /** Copy assignment is disabled. */
    ConfigIter& operator = (const ConfigIter&) = delete;

    /** Return true if the object was built by ConfigIter(). */
    bool default_constructor() const {return _ctor_type==CTOR_DEFAULT;}
    /** Return true if the object was built by ConfigIter(Xtc*, const void*). */
    bool regular_constructor() const {return _ctor_type==CTOR_REGULAR;}
    /** Return the CTOR_TYPE recorded by the constructor. */
    CTOR_TYPE constructor_type() const {return _ctor_type;}

private:
    std::unordered_map<unsigned,ShapesData*> _shapesData;
    DescData* _desc_shape = NULL;
    DescData* _desc_value = NULL;
    CTOR_TYPE _ctor_type;
};
};

#endif //

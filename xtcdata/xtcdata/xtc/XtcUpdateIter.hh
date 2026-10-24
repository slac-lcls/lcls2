/**
 * @file
 * @brief Declares XtcData::DataDef (a list of field definitions with a name index) and XtcData::XtcUpdateIter, an iterator that copies, filters and builds xtcs.
 */
#ifndef XTCDATA_DEBUGITER_H
#define XTCDATA_DEBUGITER_H

/*
 * class XtcUpdateIter provides access to all types of xtc
 */

#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"

#include <iostream>
#include <map>
#include <iterator>
#include <string>
#include <typeinfo>
#include <memory>

namespace XtcData
{

/* Keeps a triplet Name vector and a lookup _index map
      NameVec: stores triplet `name`, `dtype`, and `rank`
      _index:  maps element no. (in the order it was added)
*/
/** VarDef that also keeps a map from field name to the order in which it was added. */
class DataDef : public XtcData::VarDef
{
public:
    /** Construct an empty definition list. */
    DataDef()
    {
        _n_elems = 0;
    }

    /**
     * Append a Name(name, (Name::DataType)dtype, rank) to NameVec and map name to its position. A repeated name keeps its first position in the map.
     * The Name constructor aborts on an invalid name or rank.
     */
    void add(char* name, unsigned dtype, int rank){
        Name::DataType dt = (Name::DataType) dtype;
        NameVec.push_back({name, dt, rank});
        std::string s(name);
        _index.insert(std::pair<std::string, int>(s, _n_elems));
        _n_elems++;
    }

    /** Print "List of names" and each name, then "List of indices" and each name with its position, to stdout. */
    void show() {
        printf("List of names\n");
        for (auto i=NameVec.begin(); i!=NameVec.end(); ++i)
            std::cout << i->name() << std::endl;
        printf("List of indices\n");
        std::map<std::string, int>::iterator itr;
        for (itr = _index.begin(); itr != _index.end(); ++itr){
            std::cout << '\t' << itr->first << '\t' << itr->second << '\n';
        }
    }

    /** Return the position recorded for name, or -1 if it is not found. */
    int index(char* name) {
        // Locates name index using name in datadef
        // TODO: Add check for newIndex >= 0
        std::string s(name);
        for (auto itr = _index.find(s); itr!=_index.end(); itr++){
            //std::cout << "DataDef.index " << itr->first << '\t' << itr->second << '\n';
            return itr->second;
        }
        return -1;
    }

    /** Return the DataType of the field called name. A missing name is not checked (index -1 is used). */
    int getDtype(char* name) {
        // Returns corresponding dtype
        int foundIndex = index(name);
        Name foundName = NameVec[foundIndex];
        return foundName.type();
    }

    /** Return the rank of the field called name. A missing name is not checked (index -1 is used). */
    int getRank(char* name) {
        // Returns corresponding rank
        int foundIndex = index(name);
        Name foundName = NameVec[foundIndex];
        return foundName.rank();
    }

private:
    std::map<std::string, int> _index;
    int _n_elems;

}; // end class DataDef

/**
 * XtcIterator that copies Names and ShapesData xtcs into an output buffer (setOutput()), dropping ShapesData whose detName_algName key is filtered, and records Names in its own NamesLookup.
 * It also has helpers to add Names and ShapesData and to create or edit Dgram headers.
 */
class XtcUpdateIter : public XtcData::XtcIterator
{
public:
    /** Return values of process(). XtcIterator::iterate() stops when process() returns 0 (Stop). */
    enum {Stop, /**< Value 0: stop iterating. */ Continue /**< Value 1: keep iterating. */ };

    /** Store numWords (passed by get_value() to the array dump helper) and zero the counters and flags; the NamesId range trackers start at 0 and 255. The output buffer is not set. */
    XtcUpdateIter(unsigned numWords) : XtcData::XtcIterator(), _numWords(numWords) {
        _bufSize = 0;
        _payloadSize = 0;
        _removedSize = 0;              // counting size of removed det/alg in bytes
        _cfgFlag = 0;                   // tells if this dgram is a Configure
        _cfgWriteFlag = 0;              // default is not to write to _cfgbuf when iterated.
        _nodeId = 0;
        _maxOfMinNamesId = 0;           // stores the highest value of the lower range existing NamesIds
        _minOfMaxNamesId = 255;         // stores the lowest value of the upper range existing NamesIds
    }

    /** Destructor; does nothing. */
    ~XtcUpdateIter() {
    }

    /**
     * Iterate into Parent xtcs. For Names: store a NameIndex, add its detName_algName filter key (flag 0), record its node id, update the NamesId range trackers, and copy it to the output buffer if the cfg-write flag is 1.
     * For ShapesData (throws a const char* if its NamesId has no Names): if the cfg flag is 1, copy it only when the cfg-write flag is 1; otherwise copy it unless its key is filtered, in which case add its size to the removed size.
     * @return Always Continue.
     */
    virtual int process(XtcData::Xtc* xtc, const void* bufEnd);

    /**
     * Print field i of descdata to stdout: scalars as 'name': value, CHARSTR arrays as a quoted string, and FLOAT16 arrays as 'name': N fp16 elements (N is the first dimension).
     * ENUMDICT arrays print a warning line; other arrays go to a dump helper that prints nothing because VERBOSE is 0 in XtcUpdateIter.cc.
     */
    void get_value(int i, Name& name, DescData& descdata);

    /** Return the size set by the last copyParent(): sizeof(Dgram) plus the bytes copied to the output buffer before that call. */
    unsigned getSize(){
        return _bufSize;
    }

    /** Return the bytes of filtered ShapesData skipped by process() since the last copyParent(). */
    uint32_t getRemovedSize(){
        return _removedSize;
    }

    /** Return the node id of the last Names xtc seen by process() (0 initially). */
    unsigned getNodeId(){
        return _nodeId;
    }

    /**
     * Increment the lower-range NamesId tracker and return its new value.
     * Prints a message and throws a const char* if the new value equals the upper-range tracker.
     */
    unsigned getNextNamesId(){
        // Returns the next available NamesId from the the maximum
        // value of the minimum range. If th next value clashes with
        // the min value of the max range, exit.
        unsigned nextNamesId = _maxOfMinNamesId + 1;

        // Update max value of the lower range
        _maxOfMinNamesId = nextNamesId;

        if (nextNamesId == _minOfMaxNamesId) {
            printf("*** NamesId full: next namesid %u not available\n", nextNamesId);
            throw "unavailable namesid";
        }
        return nextNamesId;
    }

    /** Set the cfg flag. When it is 1, process() copies ShapesData only if the cfg-write flag is 1 and ignores filters; the member comment says it marks a Configure datagram. */
    void setCfgFlag(int cfgFlag) {
        _cfgFlag = cfgFlag;
    }
    /** Set the cfg-write flag. When it is 1, process() copies Names xtcs (and, with the cfg flag, ShapesData) to the output buffer. */
    void setCfgWriteFlag(int cfgWriteFlag) {
        _cfgWriteFlag = cfgWriteFlag;
    }
    /** Set the output buffer. copyPayload() writes after its first sizeof(Dgram) bytes and copyParent() writes the Dgram header at its start. */
    void setOutput(char* outbuf) {
        _outbuf = outbuf;
    }

    /** Return the cfg flag set by setCfgFlag(). */
    int isConfig(){
        return _cfgFlag;
    }

    /**
     * Append to xtc a Names xtc built from detName, detType, detId, segment, NamesId(nodeId, namesId) and Alg(algName, major, minor, micro), add one Name per datadef entry, and store its NameIndex in the internal NamesLookup.
     * The Names/Name constructors and Xtc::alloc() abort on invalid names or if bufEnd would be passed.
     */
    void addNames(Xtc& xtc, const void* bufEnd, char* detName, char* detType, char* detId,
            unsigned nodeId, unsigned namesId, unsigned segment,
            char* algName, uint8_t major, uint8_t minor, uint8_t micro,
            DataDef& datadef);
    /** Call set_string() on the CreateData from the last createData() call for field datadef.index(varname), with data as the string. */
    void setString(char* data, DataDef& datadef, char* varname);
    /**
     * Read a scalar of the field's DataType from data and append it with set_value() to the CreateData from the last createData() call.
     * The field is datadef.index(varname) in the Names for NamesId(nodeId, namesId); CHARSTR, ENUMVAL, ENUMDICT and FLOAT16 only print a "not handled" message.
     */
    void setValue(unsigned nodeId, unsigned namesId,
            char* data, DataDef& datadef, char* varname);
    /**
     * Allocate array field datadef.index(varname) with the given shape in the current CreateData (allocate<T>()) and copy Shape(shape).size(name) bytes from data into it.
     * T follows the field's DataType in the Names for NamesId(nodeId, namesId); ENUMVAL, ENUMDICT and FLOAT16 only print a "not handled" message.
     */
    void addData(unsigned nodeId, unsigned namesId,
            unsigned* shape, char* data, DataDef& datadef, char* varname);
    /**
     * Placement-construct at buf a Dgram of type Event with transition id transId, env low bits 0 and an empty Parent xtc, and return it.
     * The time is TimeStamp(timestamp_val) if counting_timestamps; otherwise gettimeofday() seconds, with the microsecond count stored unscaled in the nanoseconds word.
     */
    Dgram& createTransition(unsigned transId, bool counting_timestamps,
                        uint64_t timestamp_val, char* buf);
    /**
     * Create a CreateData for NamesId(nodeId, namesId) that appends a new ShapesData to xtc, using the internal NamesLookup, and keep it in a unique_ptr.
     * Replacing a previous CreateData runs its destructor, which aborts if it did not receive all its fields.
     */
    void createData(Xtc& xtc, const void* bufEnd, unsigned nodeId, unsigned namesId);
    /** Set d.time to TimeStamp(timestamp_val): low 32 bits nanoseconds, high 32 bits seconds. */
    void updateTimeStamp(Dgram& d, uint64_t timestamp_val);
    /** Replace bits 27:24 of d.env with transtionId, which is shifted left by 24 and ORed in, so values above 15 also set bits in 31:28. */
    void updateService(Dgram& d, uint8_t transtionId);
    /** Set d.xtc.damage to damage. */
    void updateDamage(Dgram& d, uint16_t damage);
    /** Return Name::get_element_size() of field datadef.index(varname) in the Names for NamesId(nodeId, namesId). */
    int getElementSize(unsigned nodeId, unsigned namesId,
            DataDef& datadef, char* varname);
    /**
     * Copy the Dgram header parent_d to the start of the output buffer, set getSize() to sizeof(Dgram) plus the bytes copied so far, reset the copied and removed byte counts, and clear all filter flags.
     * The copied header's extent is not changed.
     */
    void copyParent(Dgram* parent_d);
    /** Copy in_size bytes from in_buf into the output buffer after sizeof(Dgram) plus the bytes already copied, and add in_size to that count. There is no bounds check. */
    void copyPayload(char* in_buf, unsigned in_size);
    /** Set the filter flag of the existing key detName_algName to 1, so process() drops matching ShapesData; unknown keys are ignored. */
    void setFilter(char* detName, char* algName);
    /** Reset all filter flags to 0. */
    void clearFilter();

private:
    NamesLookup _namesLookup;
    unsigned _numWords;
    std::unique_ptr<CreateData> _newData;

    // The _outbuf is used for storing Names and ShapesData
    // while they are being iterated (copy if no filter matched).
    // Note that Names and ShapesData are copied to _outbuf after
    // sizeof(Dgram) offset. This gap is reserved for the parent
    // dgram that will get copied when save() is called and the
    // new extent has been calculated (if data were removed).
    // For Configure, it's first iterated to get NodeId and (next) NamesId
    // then iterated again after all Names have been added for writing
    // to _outbuf. Caller has to set _cfgWriteFlag for writing.
    unsigned _payloadSize;
    unsigned _bufSize;
    char* _outbuf;

    // Used for couting no. of ShapesData bytes removed per event.
    // This gets reset to 0 when the event is saved.
    uint32_t _removedSize;

    // Used for storing detName_algName (key) and its per-event
    // filter flag. 0 (initial values) means keeps while 1 means
    // filtered. This map gets reset to 0 when an event is saved.
    std::map<std::string, int> _flagFilter;

    // Used for checking if this is a Configure dgram and allowing
    // writing to _cfgbuf when iterated.
    int _cfgFlag;
    int _cfgWriteFlag;

    // When Names is iterated, we keep track of NodeId and NamesId
    unsigned _nodeId;
    unsigned _maxOfMinNamesId;
    unsigned _minOfMaxNamesId;

}; // end class XtcUpdateIter


}; // end namespace XtcData

#endif //

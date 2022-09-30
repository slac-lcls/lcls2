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

#define MAXBUFSIZE 0x4000000

namespace XtcData
{

/* Keeps a triplet Name vector and a lookup _index map
      NameVec: stores triplet `name`, `dtype`, and `rank`
      _index:  maps element no. (in the order it was added)
*/
class DataDef : public XtcData::VarDef
{
public:
    DataDef()
    {
        _n_elems = 0;
    }

    void add(char* name, unsigned dtype, int rank){
        Name::DataType dt = (Name::DataType) dtype;
        NameVec.push_back({name, dt, rank});
        std::string s(name);
        _index.insert(std::pair<std::string, int>(s, _n_elems));
        _n_elems++;
    }

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

    int getDtype(char* name) {
        // Returns corresponding dtype
        int foundIndex = index(name);
        Name foundName = NameVec[foundIndex];
        return foundName.type();
    }

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

class XtcUpdateIter : public XtcData::XtcIterator
{
public:
    enum {Stop, Continue};

    XtcUpdateIter(unsigned numWords, uint64_t maxBufSize) : XtcData::XtcIterator(), _numWords(numWords) {
        // Users can pass in customized buffer's size for three buffers below
        if (maxBufSize == 0) {
            maxBufSize = MAXBUFSIZE;
        }
        _bufSize = 0;
        _payloadSize = 0;
        _removedSize = 0;              // counting size of removed det/alg in bytes
        _cfgFlag = 0;                   // tells if this dgram is a Configure
        _cfgWriteFlag = 0;              // default is not to write to _cfgbuf when iterated.
        _nodeId = 0;
        _maxOfMinNamesId = 0;           // stores the highest value of the lower range existing NamesIds
        _minOfMaxNamesId = 255;         // stores the lowest value of the upper range existing NamesIds
    }

    ~XtcUpdateIter() {
    }

    virtual int process(XtcData::Xtc* xtc, const void* bufEnd);

    void get_value(int i, Name& name, DescData& descdata);

    unsigned getSize(){
        return _bufSize;
    }

    uint32_t getRemovedSize(){
        return _removedSize;
    }

    unsigned getNodeId(){
        return _nodeId;
    }

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

    void setCfgFlag(int cfgFlag) {
        _cfgFlag = cfgFlag;
    }
    void setCfgWriteFlag(int cfgWriteFlag) {
        _cfgWriteFlag = cfgWriteFlag;
    }
    void setOutput(char* outbuf) {
        _outbuf = outbuf;
    }

    int isConfig(){
        return _cfgFlag;
    }

    void addNames(Xtc& xtc, const void* bufEnd, char* detName, char* detType, char* detId,
            unsigned nodeId, unsigned namesId, unsigned segment,
            char* algName, uint8_t major, uint8_t minor, uint8_t micro,
            DataDef& datadef);
    void setString(char* data, DataDef& datadef, char* varname);
    void setValue(unsigned nodeId, unsigned namesId,
            char* data, DataDef& datadef, char* varname);
    void addData(unsigned nodeId, unsigned namesId,
            unsigned* shape, char* data, DataDef& datadef, char* varname);
    Dgram& createTransition(unsigned transId, bool counting_timestamps,
                        uint64_t timestamp_val, const void** bufEnd, uint64_t maxBufSize);
    void createData(Xtc& xtc, const void* bufEnd, unsigned nodeId, unsigned namesId);
    void updateTimeStamp(Dgram& d, uint64_t timestamp_val);
    int getElementSize(unsigned nodeId, unsigned namesId,
            DataDef& datadef, char* varname);
    void copyParent(Dgram* parent_d);
    void copyPayload(char* in_buf, unsigned in_size);
    void setFilter(char* detName, char* algName);
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

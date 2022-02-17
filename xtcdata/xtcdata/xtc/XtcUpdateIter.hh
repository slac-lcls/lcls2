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

#define BUFSIZE 0x1000000

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

private:
    std::map<std::string, int> _index;
    int _n_elems;

}; // end class DataDef

class XtcUpdateIter : public XtcData::XtcIterator
{
public:
    enum {Stop, Continue};
    
    XtcUpdateIter(unsigned numWords) : XtcData::XtcIterator(), _numWords(numWords) {
        _bufsize = 0;
        _buf = (char *) malloc(BUFSIZE);
        _tmpbufsize = 0;
        _tmpbuf = (char *) malloc(BUFSIZE);
        _removed_size = 0; // counting size of removed det/alg in bytes
    }

    ~XtcUpdateIter() {
        free(_buf);
        free(_tmpbuf);
    }
    
    virtual int process(XtcData::Xtc* xtc);
    
    void get_value(int i, Name& name, DescData& descdata);

    char* get_buf(){
        return _buf;
    }

    unsigned get_bufsize(){
        return _bufsize;
    }

    void clear_buf(){
        _bufsize = 0;
    }

    uint32_t get_removed_size(){
        return _removed_size;
    }

    void addNames(Xtc& xtc, char* detName, char* detType, char* detId, 
            unsigned nodeId, unsigned namesId, unsigned segment,
            char* algName, uint8_t major, uint8_t minor, uint8_t micro,
            DataDef& datadef);
    void setString(char* data, DataDef& datadef, char* varname);
    void setValue(unsigned nodeId, unsigned namesId,          
            char* data, DataDef& datadef, char* varname);
    void addData(unsigned nodeId, unsigned namesId, 
            unsigned* shape, char* data, DataDef& datadef, char* varname);
    Dgram& createTransition(unsigned transId, bool counting_timestamps,
                        unsigned timestamp_val);
    void createData(Xtc& xtc, unsigned nodeId, unsigned namesId);
    void updateTimeStamp(Dgram& d, unsigned sec, unsigned nsec);
    int getElementSize(unsigned nodeId, unsigned namesId, 
            DataDef& datadef, char* varname);
    void copy(Dgram* parent_d);
    void copy2buf(char* in_buf, unsigned in_size);
    void copy2tmpbuf(char* in_buf, unsigned in_size);
    void setFilter(char* detName, char* algName);


private:
    NamesLookup _namesLookup;
    unsigned _numWords;
    std::unique_ptr<CreateData> _newData;
    
    // _tmpbuf* are used for storing Names and ShapesData
    // while they are being iterated (copy if no filter matched).
    // bu* are the main buffer that has both parent dgram
    // and Names & ShapesData. It aslo has infinite lifetime
    // until it gets cleared manually.
    char* _tmpbuf;
    unsigned _tmpbufsize;
    char* _buf;
    unsigned _bufsize;

    // Used for couting no. of ShapesData bytes removed per event.
    // This gets reset to 0 when the event is saved. 
    uint32_t _removed_size;
    
    // Used for storing detName_algName (key) and its per-event 
    // filter flag. 0 (initial values) means keeps while 1 means 
    // filtered. This map gets reset to 0 when an event is saved. 
    std::map<std::string, int> _flagFilter; 
}; // end class XtcUpdateIter


}; // end namespace XtcData

#endif //

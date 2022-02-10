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
        // Locate name index using name in datadef
        std::string s(name);
        for (auto itr = _index.find(s); itr!=_index.end(); itr++){
            std::cout << "DataDef.index " << itr->first << '\t' << itr->second << '\n';
            return itr->second;
        }
        return -1;
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
        _removed_size = 0; // counting size of removed det/alg in bytes
    }

    ~XtcUpdateIter() {
        free(_buf);
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

    void copy2buf(char* in_buf, unsigned in_size);
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


private:
    NamesLookup _namesLookup;
    unsigned _numWords;
    char* _buf;
    unsigned _bufsize;
    std::unique_ptr<CreateData> _newData;
    uint32_t _removed_size;
}; // end class XtcUpdateIter


}; // end namespace XtcData

#endif //

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

#define BUFSIZE 0x1000000

namespace XtcData
{

class NewDef : public XtcData::VarDef
{
public:
    NewDef()
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
        // Locate name index using name in newdef
        std::string s(name);
        for (auto itr = _index.find(s); itr!=_index.end(); itr++){
            std::cout << itr->first << '\t' << itr->second << '\n';
            return itr->second;
        }
        return -1;
    }

private:
    std::map<std::string, int> _index;
    int _n_elems;

}; // end class NewDef

class XtcUpdateIter : public XtcData::XtcIterator
{
public:
    enum {Stop, Continue};
    
    XtcUpdateIter(unsigned numWords) : XtcData::XtcIterator(), _numWords(numWords) {
        _bufsize = 0;
        _buf = (char *) malloc(BUFSIZE);
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

    void copy2buf(char* in_buf, unsigned in_size);
    void addNames(Xtc& xtc, char* detName, char* detType, char* detId, 
            unsigned nodeId, unsigned namesId, unsigned segment,
            char* algName, uint8_t major, uint8_t minor, uint8_t micro,
            NewDef& newdef);
    void addData(Xtc& xtc, unsigned nodeId, unsigned namesId, 
            unsigned* shape, char* data, NewDef& newdef, char* varname);


private:
    NamesLookup _namesLookup;
    unsigned _numWords;
    char* _buf;
    unsigned _bufsize;
}; // end class XtcUpdateIter


}; // end namespace XtcData

#endif //

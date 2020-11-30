#ifndef JSON2XTC__H
#define JSON2XTC__H
#include "rapidjson/filereadstream.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include <vector>
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/DescData.hh"
#include <Python.h>

namespace Pds
{
    class JsonIterator {
    public:
        static std::map<std::string, enum XtcData::Name::DataType> typeMap;
        JsonIterator(rapidjson::Value &root, rapidjson::Value &types) : _root(root), _types(types) {}
        void iterate() { iterate(_root); };
        std::string curname();
        virtual void process(rapidjson::Value &val) {};
    protected:
        void iterate(rapidjson::Value &val);
        rapidjson::Value *findJsonType();
    private:
        rapidjson::Value &_root;
        rapidjson::Value &_types;
        std::vector<std::string> _names;
        std::vector<bool>        _isnum;
    };

    int translateJson2Xtc(char *in, char *out, XtcData::NamesId namesID, const char* detname=0, unsigned segment=0);
    int translateJson2Xtc( PyObject* item, XtcData::Xtc& xtc, XtcData::NamesId namesID);
    int translateJson2XtcNames(rapidjson::Document* d, 
                               XtcData::Xtc* xtc, 
                               XtcData::NamesLookup& nl, 
                               XtcData::NamesId namesID, 
                               rapidjson::Value& json, 
                               const char* detname, 
                               unsigned segment);
    int translateJson2XtcData (rapidjson::Document* d, 
                               XtcData::Xtc* xtc, 
                               XtcData::NamesLookup& nl, 
                               XtcData::NamesId namesID, 
                               rapidjson::Value& json);

}; // namespace Pds

#endif // JSON2XTC__H

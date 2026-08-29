#ifndef JSON2XTC__H
#define JSON2XTC__H
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include <python3.11/Python.h>
#include <string>
#include <vector>

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

    int translateJson2Xtc(char *in,
                          char *out,
                          const void* bufEnd,
                          XtcData::NamesId namesID,
                          const char* detname=0,
                          unsigned segment=0);
    int translateJson2Xtc(PyObject* item,
                          XtcData::Xtc& xtc,
                          const void* bufEnd,
                          XtcData::NamesId namesID);
    int translateJson2Xtc(PyObject* item,
                          XtcData::Xtc& xtc,
                          const void* bufEnd,
                          XtcData::NamesId namesID,
                          unsigned segment,
                          std::string serNo=std::string(""));
    int translateJson2XtcNames(rapidjson::Document* d,
                               XtcData::Xtc* xtc,
                               const void* bufEnd,
                               XtcData::NamesLookup& nl,
                               XtcData::NamesId namesID,
                               rapidjson::Value& json,
                               const char* detname,
                               unsigned segment,
                               std::string const& serNo = std::string(""));
                               //const char* serNo=std::string("").c_str());
    int translateJson2XtcData (rapidjson::Document* d,
                               XtcData::Xtc* xtc,
                               const void* bufEnd,
                               XtcData::NamesLookup& nl,
                               XtcData::NamesId namesID,
                               rapidjson::Value& json);

}; // namespace Pds

#endif // JSON2XTC__H

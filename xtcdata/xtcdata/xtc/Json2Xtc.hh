#ifndef JSON2XTC__H
#define JSON2XTC__H
#include "rapidjson/filereadstream.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
//#include <string.h>
#include <vector>
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/DescData.hh"

using namespace rapidjson;

namespace XtcData
{
    class JsonIterator {
    public:
        static std::map<std::string, enum Name::DataType> typeMap;
        JsonIterator(Value &root, Value &types) : _root(root), _types(types) {}
        void iterate() { iterate(_root); };
        std::string curname();
        virtual void process(Value &val) {};
    protected:
        void iterate(Value &val);
        Value *findJsonType();
    private:
        Value &_root;
        Value &_types;
        std::vector<std::string> _names;
        std::vector<bool>        _isnum;
    };

    int translateJson2Xtc(char *in, char *out, NamesId namesID);

}; // namespace XtcData

#endif // JSON2XTC__H

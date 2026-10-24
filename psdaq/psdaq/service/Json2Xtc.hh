/**
 * @file
 * @brief Translation of configuration JSON (with :types: and :enum: descriptions) into XTC Names and data.
 */
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
//#include <python3.11/Python.h>
#include <python3.9/Python.h>
#include <string>
#include <vector>

namespace Pds
{
    /** Walks a JSON value depth-first (object members in sorted order) and calls process() for each leaf, keeping the dotted name path; a parallel types value describes the leaves. */
    class JsonIterator {
    public:
        /** Map from type name strings (UINT8, ..., DOUBLE, CHARSTR, ENUMVAL, ENUMDICT) to XtcData::Name data types. */
        static std::map<std::string, enum XtcData::Name::DataType> typeMap;
        /** Keep references to the value root to walk and to the matching types description. */
        JsonIterator(rapidjson::Value &root, rapidjson::Value &types) : _root(root), _types(types) {}
        /** Walk the root value, calling process() for every scalar and every array that contains no objects or arrays. */
        void iterate() { iterate(_root); };
        /** Return the current path: names joined with dots, and array indices appended with underscores. */
        std::string curname();
        /** Hook called for each leaf; the default does nothing. */
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

    /** Parse the JSON text in, create a Parent Xtc at out (bounded by bufEnd) and add the Names and data for it with namesID, detname (if non-null) and segment. Returns the Xtc extent, or -1 if the Names translation fails. */
    int translateJson2Xtc(char *in,
                          char *out,
                          const void* bufEnd,
                          XtcData::NamesId namesID,
                          const char* detname=0,
                          unsigned segment=0);
    /** Translate the JSON string object item into Names and data appended to xtc, taking the detector name and segment from the detName:RO entry (name_N). Returns 0, or -1 if detName:RO is missing or has no _N suffix or a translation step fails. */
    int translateJson2Xtc(PyObject* item,
                          XtcData::Xtc& xtc,
                          const void* bufEnd,
                          XtcData::NamesId namesID);
    /** Like translateJson2Xtc(item, xtc, bufEnd, namesID), but uses serNo as detector ID when it is non-empty. The segment argument is overwritten by the number parsed from detName:RO. */
    int translateJson2Xtc(PyObject* item,
                          XtcData::Xtc& xtc,
                          const void* bufEnd,
                          XtcData::NamesId namesID,
                          unsigned segment,
                          std::string serNo=std::string(""));
    /** Check that the document has alg:RO, detName:RO, detType:RO, detId:RO and doc:RO, add Names (detector name, algorithm, type, serNo or detId:RO, namesID, segment) with one entry per leaf typed from :types: (plus ENUMDICT entries for :enum:) to xtc, register them in nl, and move :types: into json. Those members are removed from the document. Returns 0, or -1 on a parse error or missing field. */
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
    /** Write the values of the remaining document members (and the :enum: dictionary values first) into xtc as data for namesID, typed from json; returns 0. */
    int translateJson2XtcData (rapidjson::Document* d,
                               XtcData::Xtc* xtc,
                               const void* bufEnd,
                               XtcData::NamesLookup& nl,
                               XtcData::NamesId namesID,
                               rapidjson::Value& json);

}; // namespace Pds

#endif // JSON2XTC__H

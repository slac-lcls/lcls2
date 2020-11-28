#include "Json2Xtc.hh"

#include <list>
#include <string>

using namespace XtcData;
using namespace rapidjson;

#include "psalg/utils/SysLog.hh"
using logging = psalg::SysLog;

namespace Pds {
std::map<std::string, enum Name::DataType> JsonIterator::typeMap = {
    {"UINT8",    Name::UINT8},
    {"UINT16",   Name::UINT16},
    {"UINT32",   Name::UINT32},
    {"UINT64",   Name::UINT64},
    {"INT8",     Name::INT8},
    {"INT16",    Name::INT16},
    {"INT32",    Name::INT32},
    {"INT64",    Name::INT64},
    {"FLOAT",    Name::FLOAT},
    {"DOUBLE",   Name::DOUBLE},
    {"CHARSTR",  Name::CHARSTR},
    {"ENUMVAL",  Name::ENUMVAL},
    {"ENUMDICT", Name::ENUMDICT}
};


static std::list<std::string> sorted_list(Value& val) {
    std::list<std::string> members;
    for (Value::MemberIterator itr = val.MemberBegin();
         itr != val.MemberEnd();
         ++itr) {
        members.push_back(std::string(itr->name.GetString()));
    }
    members.sort();
    return members;
}


void JsonIterator::iterate(Value &val) {
    if (val.IsObject()) {
        std::list<std::string> members = sorted_list(val);
        for (auto itr = members.begin();
             itr != members.end(); itr++) {
            _names.push_back(*itr);
            _isnum.push_back(false);
            iterate(val[itr->c_str()]);
            _names.pop_back();
            _isnum.pop_back();
        }
    } else if (val.IsArray()) {
        // Is this a simple array? i.e. No arrays or objects!
        int cnt;
        Value::ValueIterator itr;
        for (cnt = 0, itr = val.Begin(); itr != val.End(); ++cnt, ++itr) {
            Value &map = val[cnt];
            if (map.IsObject() || map.IsArray())
                break;
        }
        if (itr == val.End()) {  // Yes, no objects or arrays!
            process(val);
        } else {
            for (cnt = 0, itr = val.Begin(); itr != val.End(); ++cnt, ++itr) {
                _names.push_back(std::to_string(cnt));
                _isnum.push_back(true);
                iterate(val[cnt]);
                _names.pop_back();
                _isnum.pop_back();
            }
        }
    } else {
        process(val);
    }
}

std::string JsonIterator::curname() {
    std::string result = "";
    for (unsigned i = 0; i < _names.size(); i++) {
        if (result == "")
            result += _names[i];
        else if (_isnum[i])
            result += "_" + _names[i];
        else
            result += "." + _names[i];
    }
    return result;
}

Value *JsonIterator::findJsonType() {
    Value *typ = &_types;
    for (unsigned i = 0; i < _names.size(); i++) {
        if (!_isnum[i])
            typ = &((*typ)[_names[i].c_str()]);
    }
    return typ;
}

class JsonFindArrayIterator : public JsonIterator
{
public:
    JsonFindArrayIterator(Value &root, Value &types, VarDef &vars)
        : JsonIterator(root, types), _vars(vars) {};
    void process(Value &val) {
        std::string name = curname();
        Value *typ = findJsonType();
        if (typ->IsArray()) {
            std::string s = (*typ)[0].GetString();
            int cnt = (int)((*typ).Size()) - 1;
            if (typeMap.find(s) == typeMap.end()) {
                _vars.NameVec.push_back({(name + ":" + s).c_str(), Name::ENUMVAL, cnt});
            } else
                _vars.NameVec.push_back({name.c_str(), typeMap[s], cnt});
        } else {
            std::string s = typ->GetString();
            if (typeMap.find(s) == typeMap.end()) {
                _vars.NameVec.push_back({(name + ":" + s).c_str(), Name::ENUMVAL});
            } else {
                Name::DataType t = typeMap[s];
                if (t == Name::CHARSTR)
                    _vars.NameVec.push_back({name.c_str(), t, 1});
                else
                    _vars.NameVec.push_back({name.c_str(), t});
            }
        }
    }
private:
    VarDef &_vars;
};

class JsonCreateDataIterator : public JsonIterator
{
public:
    JsonCreateDataIterator(Value &root, Value &types, CreateData &cd)
        : JsonIterator(root, types), _cd(cd), _cnt(0) {};
    template <typename T> T getVal(Value &val, Name::DataType typ) {
        switch (typ) {
        case Name::UINT8:
        case Name::UINT16:
        case Name::UINT32:
            return (T) val.GetUint();
            break;
        case Name::UINT64:
            return (T) val.GetUint64();
            break;
        case Name::INT8:
        case Name::INT16:
        case Name::INT32:
        case Name::ENUMVAL:
            return (T) val.GetInt();
            break;
        case Name::INT64:
            return (T) val.GetInt64();
            break;
        case Name::FLOAT:
            return (T) val.GetFloat();
            break;
        case Name::DOUBLE:
            return (T) val.GetDouble();
            break;
        default:
            /* Don't support these. */
            break;
        }
        // cpo: I think this shouldn't happen, but this avoids
        // compiler warnings.
        return (T)0;
    }
    template <typename T> void writeArray(Value &val, unsigned shape[MaxRank],
                                          unsigned size, Name::DataType typ) {
        Array<T> arrayT = _cd.allocate<T>(_cnt, shape);
        T *data = arrayT.data();
        for (unsigned i = 0; i < size; i++)
            data[i] = getVal<T>(val[i], typ);
    }
    void process(Value &val) {
        Value *typ = findJsonType();
        if (typ->IsArray()) {
            unsigned i;
            unsigned shape[MaxRank], size = 1;
            for (i = 1; i < (*typ).Size(); i++) {
                unsigned dim = (*typ)[i].GetUint();
                shape[i-1] = dim;
                size *= dim;
            }
            std::string s = (*typ)[0].GetString();
            Name::DataType dtyp = (typeMap.find(s) == typeMap.end()) ? Name::ENUMVAL : typeMap[s];
            switch (dtyp) {
            case Name::UINT8:
                writeArray<uint8_t>(val, shape, size, dtyp);
                break;
            case Name::UINT16:
                writeArray<uint16_t>(val, shape, size, dtyp);
                break;
            case Name::UINT32:
                writeArray<uint32_t>(val, shape, size, dtyp);
                break;
            case Name::UINT64:
                writeArray<uint64_t>(val, shape, size, dtyp);
                break;
            case Name::INT8:
                writeArray<int8_t>(val, shape, size, dtyp);
                break;
            case Name::INT16:
                writeArray<int16_t>(val, shape, size, dtyp);
                break;
            case Name::INT32:
                writeArray<int32_t>(val, shape, size, dtyp);
                break;
            case Name::INT64:
                writeArray<int64_t>(val, shape, size, dtyp);
                break;
            case Name::FLOAT:
                writeArray<float>(val, shape, size, dtyp);
                break;
            case Name::DOUBLE:
                writeArray<double>(val, shape, size, dtyp);
                break;
            case Name::CHARSTR:
                printf("Charstr array?!?\n");
                break;
            case Name::ENUMVAL:
                writeArray<int32_t>(val, shape, size, dtyp);
                break;
            case Name::ENUMDICT:
                printf("Enum dictionary?!?\n");
                break;
            }
        } else {
            std::string s = typ->GetString();
            Name::DataType dtyp = (typeMap.find(s) == typeMap.end()) ? Name::ENUMVAL : typeMap[s];
            switch (dtyp) {
            case Name::UINT8:
                _cd.set_value(_cnt, (uint8_t) val.GetUint());
                break;
            case Name::UINT16:
                _cd.set_value(_cnt, (uint16_t) val.GetUint());
                break;
            case Name::UINT32:
                _cd.set_value(_cnt, (uint32_t) val.GetUint());
                break;
            case Name::UINT64:
                _cd.set_value(_cnt, val.GetUint64());
                break;
            case Name::INT8:
                _cd.set_value(_cnt, (int8_t) val.GetInt());
                break;
            case Name::INT16:
                _cd.set_value(_cnt, (int16_t) val.GetInt());
                break;
            case Name::INT32:
                _cd.set_value(_cnt, (int32_t) val.GetInt());
                break;
            case Name::INT64:
                _cd.set_value(_cnt, val.GetInt64());
                break;
            case Name::FLOAT:
                _cd.set_value(_cnt, val.GetFloat());
                break;
            case Name::DOUBLE:
                _cd.set_value(_cnt, val.GetDouble());
                break;
            case Name::CHARSTR:
                _cd.set_string(_cnt, val.GetString());
                break;
            case Name::ENUMVAL:
                _cd.set_value(_cnt, (int32_t) val.GetInt());
                break;
            case Name::ENUMDICT:
                printf("Enum dictionary?!?\n");
                break;
            }
        }
        _cnt++;
    }

    void set_value(int32_t v) {
        _cd.set_value(_cnt, v);
        _cnt++;
    }

private:
    CreateData  &_cd;
    int          _cnt;
};

int translateJson2XtcNames(Document* d, Xtc* xtc, NamesLookup& nl, NamesId namesID, Value& json, const char* detname, unsigned segment)
{
    if (d->HasParseError()) {
        printf("Parse error: %s, location %zu\n",
               GetParseError_En(d->GetParseError()), d->GetErrorOffset());
        return -1;
    }
    if (!d->IsObject()) {
        printf("Document is not an object!\n");
        return -1;
    }
    if (!d->HasMember("alg:RO") || !d->HasMember("detName:RO") || !d->HasMember("detType:RO") ||
        !d->HasMember("detId:RO") || !d->HasMember("doc:RO")) {
        printf("Document is missing a mandatory field (alg, detName, detType, detId, or doc)!\n");
        return -1;
    }
    const Value& a = (*d)["alg:RO"];
    const Value& v = a["version:RO"];
    Alg alg = Alg(a["alg:RO"].GetString(), v[0].GetInt(), 
                  v[1].GetInt(), v[2].GetInt());
    // Set alg._doc from a["doc"].GetString()?
    d->RemoveMember("alg:RO");
    Names& names = *new(xtc) Names(detname,
                                   alg,
                                   (*d)["detType:RO"].GetString(),
                                   (*d)["detId:RO"].GetString(), namesID, segment);
    // Set _NameInfo.doc from d["doc"].GetString()?
    d->RemoveMember("detName:RO");
    d->RemoveMember("detType:RO");
    d->RemoveMember("detId:RO");
    d->RemoveMember("doc:RO");
    Value &jsv = (*d)[":types:"];
    json = jsv;              // This makes d[':types:'] null!!
    d->RemoveMember(":types:");

    VarDef vars;
    if (json.HasMember(":enum:")) {
        Value &etypes = json[":enum:"];
        std::list<std::string> members = sorted_list(etypes);
        for (auto itr = members.begin();
            itr != members.end(); itr++) {
            std::list<std::string> mem2 = sorted_list(etypes[itr->c_str()]);
            for (auto itr2 = mem2.begin();
                 itr2 != mem2.end();
                 ++itr2) {
                vars.NameVec.push_back({(*itr2 + ":" + *itr).c_str(), Name::ENUMDICT});
            }
        }
    }
    JsonFindArrayIterator fai = JsonFindArrayIterator(*d, json, vars);
    fai.iterate();
    names.add(*xtc, vars);
    nl[namesID] = NameIndex(names);

    return 0;
}

int translateJson2XtcData(Document* d, Xtc* xtc, NamesLookup& nl, NamesId namesID, Value& json)
{
    CreateData cd(*xtc, nl, namesID);
    JsonCreateDataIterator cdi = JsonCreateDataIterator(*d, json, cd);
    if (json.HasMember(":enum:")) {
        Value &etypes = json[":enum:"];
        std::list<std::string> mem = sorted_list(etypes);
        for (auto itr = mem.begin();
             itr != mem.end();
             ++itr) {
            Value &map = etypes[itr->c_str()];
            std::list<std::string> mem2 = sorted_list(map);
            for (auto itr2 = mem2.begin();
                 itr2 != mem2.end();
                 ++itr2) {
                cdi.set_value(map[itr2->c_str()].GetInt());
            }
        }
    }
    cdi.iterate();
    return 0;
}

//
// Translate a buffer containing JSON (in) to a buffer containing an
// Xtc2 structure (out) with the specified NamesId.
//
// This returns the size of what is written into out, or -1 if there
// is an error.
//
int translateJson2Xtc(char *in, char *out, NamesId namesID, const char* detname, unsigned segment)
{
    TypeId tid(TypeId::Parent, 0);
    Xtc *xtc = new (out) Xtc(tid);

    Document *d = new Document();
    NamesLookup nl;
    Value json;

    d->Parse(in);
    if (translateJson2XtcNames(d, xtc, nl, namesID, json, detname, segment) < 0)
        return -1;

    if (translateJson2XtcData (d, xtc, nl, namesID, json) < 0)
        return -1;

    delete d;
    return xtc->extent;
}

} // namespace Pds

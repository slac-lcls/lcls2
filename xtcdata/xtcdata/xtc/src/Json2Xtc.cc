#include "xtcdata/xtc/Json2Xtc.hh"

namespace XtcData {
std::map<std::string, enum Name::DataType> JsonIterator::typeMap = {
    {"UINT8",  Name::UINT8},
    {"UINT16", Name::UINT16},
    {"UINT32", Name::UINT32},
    {"UINT64", Name::UINT64},
    {"INT8",   Name::INT8},
    {"INT16",  Name::INT16},
    {"INT32",  Name::INT32},
    {"INT64",  Name::INT64},
    {"FLOAT",  Name::FLOAT},
    {"DOUBLE", Name::DOUBLE}
};

void JsonIterator::iterate(Value &val) {
    if (val.IsObject()) {
        for (Value::MemberIterator itr = val.MemberBegin();
             itr != val.MemberEnd();
             ++itr) {
            const char *name = itr->name.GetString();
            Value &map = val[name];
            _names.push_back((std::string)name);
            _isnum.push_back(false);
            iterate(map);
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
        if (itr == val.End()) {
            process(val);
        } else {
            for (cnt = 0, itr = val.Begin(); itr != val.End(); ++cnt, ++itr) {
                Value &map = val[cnt];
                _names.push_back(std::to_string(cnt));
                _isnum.push_back(true);
                iterate(map);
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
    for (int i = 0; i < _names.size(); i++) {
        if (result == "" || _isnum[i])
            result += _names[i];
        else
            result += "_" + _names[i];
    }
    return result;
}

Value *JsonIterator::findJsonType() {
    Value *typ = &_types;
    for (int i = 0; i < _names.size(); i++) {
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
            _vars.NameVec.push_back({name.c_str(),
                                     typeMap[(*typ)[0].GetString()],
                                     (int)((*typ).Size()) - 1});
        } else {
            _vars.NameVec.push_back({name.c_str(),
                                     typeMap[typ->GetString()]});
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
        }
    }
    template <typename T> void writeArray(Value &val, unsigned shape[MaxRank],
                                          unsigned size, Name::DataType typ) {
        Array<T> arrayT = _cd.allocate<T>(_cnt, shape);
        T *data = arrayT.data();
        for (int i = 0; i < size; i++)
            data[i] = getVal<T>(val[i], typ);
    }
    void process(Value &val) {
        Value *typ = findJsonType();
        if (typ->IsArray()) {
            int i;
            unsigned shape[MaxRank], size = 1;
            for (i = 1; i < (*typ).Size(); i++) {
                unsigned dim = (*typ)[i].GetUint();
                shape[i-1] = dim;
                size *= dim;
            }
            Name::DataType dtyp = typeMap[(*typ)[0].GetString()];
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
            }
        } else {
            switch (typeMap[typ->GetString()]) {
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
            }
        }
        _cnt++;
    }
private:
    CreateData  &_cd;
    int          _cnt;
};

//
// Translate a buffer containing JSON (in) to a buffer containing an
// Xtc2 structure (out) with the specified NamesId.
//
// This returns the size of what is written into out, or -1 if there
// is an error.
//
int translateJson2Xtc(char *in, char *out, NamesId namesID)
{
    TypeId tid(TypeId::Parent, 0);
    Xtc *xtc = new (out) Xtc(tid);
    Document d;
    d.Parse(in);
    if (d.HasParseError()) {
        printf("Parse error: %s, location %d\n",
               GetParseError_En(d.GetParseError()), d.GetErrorOffset());
        return -1;
    }
    if (!d.IsObject()) {
        printf("Document is not an object!\n");
        return -1;
    }
    const Value& a = d["alg"];
    const Value& v = a["version"];
    Alg alg = Alg(a["alg"].GetString(), v[0].GetInt(), 
                  v[1].GetInt(), v[2].GetInt());
    // Set alg._doc from a["doc"].GetString()?
    d.RemoveMember("alg");
    Names& names = *new(xtc) Names(d["detName"].GetString(), alg,
                                   d["detType"].GetString(),
                                   d["detId"].GetString(), namesID);
    // Set _NameInfo.doc from d["doc"].GetString()?
    d.RemoveMember("detName");
    d.RemoveMember("detType");
    d.RemoveMember("detId");
    d.RemoveMember("doc");
    Value &jsv = d["json_types"];
    Value json;
    json = jsv;              // This makes d['json_types'] null!!
    d.RemoveMember("json_types");

    VarDef vars;
    JsonFindArrayIterator fai = JsonFindArrayIterator(d, json, vars);
    fai.iterate();
    names.add(*xtc, vars);
    NamesLookup nl;
    nl[namesID] = NameIndex(names);

    CreateData cd(*xtc, nl, namesID);
    JsonCreateDataIterator cdi = JsonCreateDataIterator(d, json, cd);
    cdi.iterate();

    return xtc->extent;
}

} // namespace XtcData

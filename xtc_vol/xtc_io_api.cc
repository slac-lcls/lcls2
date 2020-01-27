#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <unistd.h>


#include <xtcdata/xtc/XtcFileIterator.hh>
#include <xtcdata/xtc/XtcIterator.hh>
#include <xtcdata/xtc/ShapesData.hh>
#include <xtcdata/xtc/DescData.hh>

#include <vector>

#include "xtc_io_api.h"

using namespace XtcData;
using namespace std;
using std::string;

typedef struct MappingLayer {
    void* xtc_root_it; //file root
    void* cur_it; //current xtc it, points to a xtc node.
    char* cur_it_name; // something like "/grp_l1/grp_l2/"
    void* iteration_stack; //
}mapping;

#define DEBUG_PRINT printf("%s():%d\n", __func__, __LINE__);
//H5 utility function prototype
//H5 name to xtc name, second
char* name_convert(const char* h5_name){
    char* xtc_name_surfix;

    return xtc_name_surfix;
}

//vector<string> _CURRENT_PATH_TOKENS;


// show name: xtc it -> h5 name string


// search/open: h5 name string -> xtc it BFS/DFS combined.

//==============================================


class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter() : XtcIterator()
    {
    }

    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();
        int data_type = name.type();
        printf("get_value() terminal data entry ==============  %d: %s rank %d, type %d\n", i, name.name(), data_rank, data_type);
        //printf("get_value() token = %s\n", name.name());
        switch(name.type()){
        case(Name::UINT8):{
            if(data_rank > 0){
                Array<uint8_t> arrT = descdata.get_array<uint8_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(Name::UINT16):{
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(Name::UINT32):{
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(Name::UINT64):{
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(Name::INT8):{
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(Name::INT16):{
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(Name::INT32):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::INT64):{
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(Name::FLOAT):{
            if(data_rank > 0){
                Array<float> arrT = descdata.get_array<float>(i);
                printf("%s: %f, %f\n",name.name(),arrT.data()[0],arrT.data()[1]);
                    }
            else{
                printf("%s: %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(Name::DOUBLE):{
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(i);
                printf("%s: %f, %f, %f\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %f\n",name.name(),descdata.get_value<double>(i));
            }
            break;
        }

        case(Name::CHARSTR):{
            if(data_rank > 0){
                Array<char> arrT = descdata.get_array<char>(i);
                printf("%s: \"%s\"\n",name.name(),arrT.data());
                    }
            else{
                printf("%s: string with no rank?!?\n",name.name());
            }
            break;
        }

        case(Name::ENUMVAL):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::ENUMDICT):{
            if(data_rank > 0){
                printf("%s: enumdict with rank?!?\n", name.name());
            } else{
                printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }
        }
    }

    int process_old(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            printf("Found TypeID == Parent, iterating...\n");
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();
        printf("===============================\n");
	    printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
                   names.detName(), names.segment(), names.detType(),
                   alg.name(), alg.version(), (int)names.namesId());
	    printf("process(): TypeId::Names: current token = %s\n", names.detName());
	    //_CURRENT_PATH_TOKENS.push_back(names.detName());
	    append_token(names.detName());
        for (unsigned i = 0; i < names.num(); i++) {
            Name& name = names.get(i);
            printf("      process() level 2 token = %s\n", name.name());
            //_CURRENT_PATH_TOKENS.push_back(name.name());
            append_token(name.name());
            print_local_path();
            printf("      Name: %s, Type: %d, Rank: %d.\n",name.name(),name.type(), name.rank());
            pop_token();
        }
        pop_token();

            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this namesid
            // may not have a NamesLookup.  cpo thinks this
            // should be fatal, since it is a sign the xtc is "corrupted",
            // in some sense.
            if (_namesLookup.count(namesId)<=0) {
                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
                throw "invalid namesid";
                break;
            }
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
            Data& data = shapesdata.data();
            printf("===============================\n");
	    printf("ShapesData group:Found %d names, the token is detName = [%s]\n",names.num(), names.detName());
	    append_token(names.detName());
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                append_token(name.name());
                print_local_path();
                get_value(i, name, descdata);
                pop_token();
            }
        pop_token();
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    int process(Xtc* xtc){
        int ret = -1;;
        switch(get_iterator_type()){
            case LIST_ALL:
                ret = process_list_all(xtc);
                break;
            case SEARCH:
                ret = process_search(xtc);
                break;
            case LIST_W_DEPTH:
                ret = process_list_with_depth(xtc);
                break;

            default:
                ret = -1;
                break;
        }
        return ret;
    }

    int process_list_all(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            printf("Found TypeID == Parent, iterating...\n");
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();
      printf("===============================\n");
      printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
                   names.detName(), names.segment(), names.detType(),
                   alg.name(), alg.version(), (int)names.namesId());
      printf("process(): TypeId::Names: current token = %s\n", names.detName());
      //_CURRENT_PATH_TOKENS.push_back(names.detName());
      append_token(names.detName());
        for (unsigned i = 0; i < names.num(); i++) {
            Name& name = names.get(i);
            printf("      process() level 2 token = %s\n", name.name());

            append_token(name.name());
            //print_local_path();
            string path_str = get_local_path();
            add_it_path(path_str);
            cout<<"Output path = " << path_str <<endl;
            //printf("      Name: %s, Type: %d, Rank: %d.\n",name.name(),name.type(), name.rank());
            pop_token();
        }
        pop_token();

            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this namesid
            // may not have a NamesLookup.  cpo thinks this
            // should be fatal, since it is a sign the xtc is "corrupted",
            // in some sense.
            if (_namesLookup.count(namesId)<=0) {
                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
                throw "invalid namesid";
                break;
            }
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
            Data& data = shapesdata.data();
            printf("===============================\n");
      printf("ShapesData group:Found %d names, the token is detName = [%s]\n",names.num(), names.detName());
      append_token(names.detName());
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                append_token(name.name());
                printf("\n");
                //print_local_path();
                string path_str = get_local_path();
                add_it_path(path_str);
                cout<<"Output path = " << path_str <<endl;
                get_value(i, name, descdata);
                pop_token();
            }
        pop_token();
            break;
        }
        default:
            break;
        }
        return Continue;
    }

    int process_list_with_depth(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            DEBUG_PRINT
            printf("Found TypeID == Parent, iterating...\n");
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            DEBUG_PRINT
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();
        //printf("===============================\n");
        //printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
        //           names.detName(), names.segment(), names.detType(),
        //           alg.name(), alg.version(), (int)names.namesId());
        string candidate_str = names.detName();
        printf("process(): TypeId::Names: current token = [%s]\n", candidate_str.c_str());
        DEBUG_PRINT
        append_token(candidate_str);
        DEBUG_PRINT
        if(1){//always list config

            printf("\n=============================== Names:: Found 2nd token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
            print_input_path();
            print_local_path();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("      process() level 2 token = %s\n", name.name());
                candidate_str = name.name();
                append_token(candidate_str);
                if(1){

                    printf("\n=============================== Terminal token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
                    print_input_path();
                    print_local_path();
                    printf("      Name: %s, Type: %d, Rank: %d.\n",name.name(),name.type(), name.rank());
                    printf("===============================\n\n");

                }
                print_local_path();

                pop_token();
            }
            printf("===============================\n\n");

        }
        pop_token();

            break;
        }
        case (TypeId::ShapesData): {
            printf("get_index() = %d, SCAN_DEPTH = %d\n", get_index(), SCAN_DEPTH);
            DEBUG_PRINT
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this namesid
            // may not have a NamesLookup.  cpo thinks this
            // should be fatal, since it is a sign the xtc is "corrupted",
            // in some sense.
            DEBUG_PRINT
            if (_namesLookup.count(namesId)<=0) {
                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
                throw "invalid namesid";
                break;
            }
            DEBUG_PRINT
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
            Data& data = shapesdata.data();
            //printf("===============================\n");
        //printf("ShapesData group:Found %d names, the token is detName = %s\n",names.num(), names.detName());

        string candidate_str = names.detName();
        append_token(candidate_str);
        //index_increment();
        if(get_index() <= SCAN_DEPTH){

            printf("\n=============================== ShapesData:: Found 2nd token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
            print_input_path();
            print_local_path();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                candidate_str = name.name();
                append_token(candidate_str);
                print_local_path();

                printf("      process() level 2 token = %s\n", name.name());
                //index_increment();
                if(get_index() <= SCAN_DEPTH){

                    printf("\n=============================== Found terminal token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
                    print_input_path();
                    print_local_path();
                    get_value(i, name, descdata); //this should be the terminal.
                    printf("===============================\n\n");

                }
                //index_decrement();
                pop_token();
            }
            printf("===============================\n\n");

        }
        //index_decrement();

        pop_token();
            break;
        }
        default:
            break;
        }
        return Continue;
    }

    int process_search(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            printf("Found TypeID == Parent, iterating...\n");
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            DEBUG_PRINT
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();
        //printf("===============================\n");
        //printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
        //           names.detName(), names.segment(), names.detType(),
        //           alg.name(), alg.version(), (int)names.namesId());
        string candidate_str = names.detName();
        printf("process(): TypeId::Names: current token = [%s]\n", candidate_str.c_str());
        append_token(candidate_str);
        index_increment();
        if(candidate_str.compare(_INPUT_PATH_TOKENS[_scan_index]) == 0){

            printf("\n=============================== Names:: Found 2nd token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
            print_input_path();
            print_local_path();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("      process() level 2 token = %s\n", name.name());
                candidate_str = name.name();
                append_token(candidate_str);
                index_increment();
                if(candidate_str.compare(_INPUT_PATH_TOKENS[_scan_index]) == 0){

                    printf("\n=============================== Terminal token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
                    print_input_path();
                    print_local_path();
                    printf("      Name: %s, Type: %d, Rank: %d.\n",name.name(),name.type(), name.rank());
                    printf("===============================\n\n");

                }
                index_decrement();
                print_local_path();

                pop_token();
            }
            printf("===============================\n\n");

        }
        index_decrement();

        pop_token();

            break;
        }
        case (TypeId::ShapesData): {
            DEBUG_PRINT
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this namesid
            // may not have a NamesLookup.  cpo thinks this
            // should be fatal, since it is a sign the xtc is "corrupted",
            // in some sense.
            DEBUG_PRINT
            if (_namesLookup.count(namesId)<=0) {
                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
                throw "invalid namesid";
                break;
            }
            DEBUG_PRINT
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
            Data& data = shapesdata.data();
            //printf("===============================\n");
        //printf("ShapesData group:Found %d names, the token is detName = %s\n",names.num(), names.detName());

        string candidate_str = names.detName();
        append_token(candidate_str);
        //index_increment();
        if(candidate_str.compare(_INPUT_PATH_TOKENS[_scan_index]) == 0){

            printf("\n=============================== ShapesData:: Found 2nd token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
            print_input_path();
            print_local_path();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                candidate_str = name.name();
                append_token(candidate_str);
                print_local_path();

                printf("      process() level 2 token = %s\n", name.name());
                //index_increment();
                if(candidate_str.compare(_INPUT_PATH_TOKENS[_scan_index]) == 0){

                    printf("\n=============================== Found terminal token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
                    print_input_path();
                    print_local_path();
                    get_value(i, name, descdata); //this should be the terminal.
                    printf("===============================\n\n");

                }
                //index_decrement();
                pop_token();
            }
            printf("===============================\n\n");

        }
        //index_decrement();

        pop_token();
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    void index_init(){
        _scan_index = -1;
    }
    int index_increment(){
        _scan_index++;
        cout<<"after increment: index = "<<_scan_index<<endl;
        return _scan_index;
    }
    int index_decrement(){
        _scan_index--;
        cout<<"after decrement: index = "<<_scan_index<<endl;
        return _scan_index;
    }
    int get_index(){
        return _scan_index;
    }
    void index_print(){
        printf("scan index = %d\n", _scan_index);
    }
    void append_token(string str){
        _CURRENT_PATH_TOKENS.push_back(str);
        index_increment();
        //cout << "After append: input token:"<< _INPUT_PATH_TOKENS[_scan_index] << ", index = "<< _scan_index <<endl;
        //index_print();
    }
    void set_input_path(vector<string> str_vec){
        _INPUT_PATH_TOKENS = str_vec;
    }
    void pop_token(){
        _CURRENT_PATH_TOKENS.pop_back();
        index_decrement();
        //cout << "After pop: input token:"<< _INPUT_PATH_TOKENS[_scan_index] << ", index = "<< _scan_index <<endl;
        //index_print();
    }
    void print_input_path(){
        //printf("input path = ");
        //_print_path(_INPUT_PATH_TOKENS);
    }
    void print_local_path(void){
        //printf("current local path = ");
        _print_path(_CURRENT_PATH_TOKENS);
//        printf("current input_path index = %d, segment = ", _scan_index);
//        for(int i = 0; i <= _scan_index; i++){
//            printf("/%s", _INPUT_PATH_TOKENS[i].c_str());
//        }
        printf("\n");
    }

    string get_local_path(){
        string ret("/");
        //_CURRENT_PATH_TOKENS
        for(int i = 0; i < _CURRENT_PATH_TOKENS.size(); i++ ){
            ret += _CURRENT_PATH_TOKENS[i];
            if(i != _CURRENT_PATH_TOKENS.size() - 1)
                ret += "/";
        }
        return ret;
    }

    int SCAN_DEPTH;

    typedef enum IteratorType{
        LIST_ALL,
        SEARCH,
        LIST_W_DEPTH,
        DEFAULT
    }ItType;

    void set_iterator_type(ItType type){
        iterator_type = type;
    }
    ItType get_iterator_type(){
        return iterator_type;
    }
    int _scan_index; //mark which token to compare

    vector<string> _INPUT_PATH_TOKENS;



    void add_it_path(string str){
        _IT_PATH_LIST.push_back(str);
    }

private:
    ItType iterator_type;
    NamesLookup _namesLookup;
    vector<string> _CURRENT_PATH_TOKENS;
    vector<string> _IT_PATH_LIST;//path list generated from xtc iterator

    void _print_path(vector<string>vec)
    {
        if(vec.size()==0)
            return;
        printf(" path = ");
        for(vector<string>::iterator it = vec.begin(); it != vec.end(); ++it){
            printf("/%s", (*it).c_str());
        }
        //printf("\n");
    }
};

vector<string> str_tok(const char* str, const char* delimiters_str){
    char* str_mod = strdup(str);
    vector<string> res;
    char * pch;
    pch = strtok(str_mod, delimiters_str);
    while (pch != NULL){
      res.push_back(string(pch));
      pch = strtok (NULL, delimiters_str);
    }
    free(str_mod);
    return res;
}

void test_str_tok(){
    char sample[] = "this is a sample string, contains/multiple.delimiters. End.";

    vector<string> v = str_tok(sample, " ,/.");
    int n = v.size();
    for(int i = 0; i < n; i++){
        cout<< v[i] << endl;
    }
}

void test_token_comp(){

}

Dgram* iterate_with_depth(int fd, int depth){

    if(depth < 1 ){
        printf("Depth must be positive.\n");
        return NULL;
    }

    Dgram* it = NULL;


    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg;
    unsigned nevent=0;
    DebugIter dbgiter;
    bool debugprint = true;
    dbgiter.index_init();
    dbgiter.set_iterator_type(DebugIter::LIST_W_DEPTH);
    dbgiter.SCAN_DEPTH = depth;
    int i =0;

    dg = iter.next();//first dg, for configure transition.
    string candidate_str = string(TransitionId::name(dg->service()));

    dbgiter.append_token(candidate_str);
    dbgiter.print_local_path();
    dbgiter.iterate(&(dg->xtc));
    dbgiter.pop_token();

    printf("\n=============================== Configure transition completed. ===============================\n");
    while ((dg = iter.next())) {//each data item in the file
        i++;
        nevent++;

        DEBUG_PRINT
        string candidate_str = string(TransitionId::name(dg->service()));
        dbgiter.append_token(candidate_str);
        if(dbgiter.get_index() <= dbgiter.SCAN_DEPTH)
            debugprint = false;
        else
            debugprint = false;
        DEBUG_PRINT
        if (debugprint) {
            dbgiter.iterate(&(dg->xtc));
        }
        //if false, dg->xtc need to move to the next block.

        DEBUG_PRINT
        dbgiter.pop_token();
    }
    return it;
}

Dgram* iterate_list_all(int fd){


    Dgram* it = NULL;


    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg;
    unsigned nevent=0;
    DebugIter dbgiter;
    bool debugprint = true;


    dbgiter.index_init();//-1
    dbgiter.set_iterator_type(DebugIter::LIST_ALL);
    int i =0;

    dg = iter.next();//first dg, for configure transition.
    string candidate_str = string(TransitionId::name(dg->service()));

    dbgiter.append_token(candidate_str);
    dbgiter.print_local_path();

    //
    dbgiter.iterate(&(dg->xtc));
    dbgiter.pop_token();

    printf("\n=============================== Configure transition completed. ===============================\n");
    while ((dg = iter.next())) {//each data item in the file
        i++;
        nevent++;

        DEBUG_PRINT
        string candidate_str = string(TransitionId::name(dg->service()));
        dbgiter.append_token(candidate_str);
        dbgiter.print_local_path();
        DEBUG_PRINT
        if (debugprint) {
            dbgiter.iterate(&(dg->xtc));
        }
        //if false, dg->xtc need to move to the next block.

        DEBUG_PRINT
        dbgiter.pop_token();
    }
    return it;

}

Dgram* target_open(int fd, const char* obj_vol_path){

    Dgram* it = NULL;
    if(!obj_vol_path)
        return it;

    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg;
    unsigned nevent=0;
    DebugIter dbgiter;
    bool debugprint = true;

    dbgiter.set_input_path(str_tok(obj_vol_path, "/ "));
    dbgiter.index_init();//set to -1;
    dbgiter.set_iterator_type(DebugIter::SEARCH);
    dbgiter.print_input_path();
    int i =0;

    dg = iter.next();//first dg, for configure transition.
    string candidate_str = string(TransitionId::name(dg->service()));

    dbgiter.append_token(candidate_str);
    dbgiter.print_local_path();
    dbgiter.iterate(&(dg->xtc));
    dbgiter.pop_token();

    printf("\n=============================== Configure transition completed. ===============================\n");
    while ((dg = iter.next())) {//each data item in the file
        i++;
        nevent++;

        DEBUG_PRINT
        string candidate_str = string(TransitionId::name(dg->service()));
        dbgiter.append_token(candidate_str);

        bool token_match = (candidate_str.compare(dbgiter._INPUT_PATH_TOKENS[dbgiter._scan_index]) == 0);

        if((candidate_str.compare(dbgiter._INPUT_PATH_TOKENS[dbgiter._scan_index]) == 0)){
            //printf("========= first section match, _scan_index = %d =========\n", dbgiter._scan_index);
            DEBUG_PRINT
            debugprint = true;
        }
        else{
            DEBUG_PRINT
            debugprint = false;
        }


        DEBUG_PRINT
        if (debugprint) {
            dbgiter.iterate(&(dg->xtc));
        }
        //if false, dg->xtc need to move to the next block.
        DEBUG_PRINT

        dbgiter.pop_token();
    }

    return it;
}

xtc_c_helper* file_open(int fd){
    Dgram* it = NULL;

    XtcFileIterator iter(fd, 0x4000000);
    Dgram* dg;
    unsigned nevent=0;
    DebugIter* dbgiter = new DebugIter();
    bool debugprint = true;


    dbgiter->index_init();//-1
    dbgiter->set_iterator_type(DebugIter::LIST_ALL);
    int i = 0;

    dg = iter.next();//first dg, for configure transition.
    string candidate_str = string(TransitionId::name(dg->service()));

    dbgiter->append_token(candidate_str);
    dbgiter->print_local_path();

    //
    dbgiter->iterate(&(dg->xtc));
    dbgiter->pop_token();

    xtc_c_helper* ret = (xtc_c_helper*)calloc(1, sizeof(xtc_c_helper));
    ret->fd = fd;
    ret->target_it = (void*)dg;
    ret->dbgiter = (void*)dbgiter;
    //ret->xtc_file_iterator = calloc(1, sizeof(XtcFileIterator));
    //memcpy(ret->xtc_file_iterator, &iter, sizeof(XtcFileIterator));

    printf("\n=============================== Configure transition completed. ===============================\n");
    while ((dg = iter.next())) {//each data item in the file
        i++;
        nevent++;

        DEBUG_PRINT
        string candidate_str = string(TransitionId::name(dg->service()));
        dbgiter->append_token(candidate_str);
        dbgiter->print_local_path();
        DEBUG_PRINT
        if (debugprint) {
            dbgiter->iterate(&(dg->xtc));
        }
        //if false, dg->xtc need to move to the next block.

        DEBUG_PRINT
        dbgiter->pop_token();
    }
    return ret;

}



EXTERNC xtc_func_t xtc_file_open(char* file_path){
    printf("xtc_file_open() is called\n");
    int fd = open(file_path, O_RDONLY);
    xtc_c_helper* ret = file_open(fd);//finished config reading
    return ret;
}

EXTERNC xtc_func_t xtc_it_open(void* param){
    xtc_c_helper* p = (xtc_c_helper*)param;

    Dgram* dg = (Dgram*) p->target_it;
    XtcFileIterator iter(p->fd, 0x4000000);
    DebugIter* dbgiter = (DebugIter*) p->dbgiter;

    while ((dg = iter.next())) {//each data item in the file
        DEBUG_PRINT
        string candidate_str = string(TransitionId::name(dg->service()));
        dbgiter->append_token(candidate_str);
        dbgiter->print_local_path();
        DEBUG_PRINT
        if (true) {
            dbgiter->iterate(&(dg->xtc));
        }
        dbgiter->pop_token();
    }

    return NULL;
}

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}


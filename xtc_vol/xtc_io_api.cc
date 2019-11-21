#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <unistd.h>


#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"

#include <vector>

using namespace XtcData;
using namespace std;
using std::string;

typedef struct MappingLayer {
    void* xtc_root_it; //file root
    void* cur_it; //current xtc it, points to a xtc node.
    char* cur_it_name; // something like "/grp_l1/grp_l2/"
    void* iteration_stack; //
}mapping;

#define DEBUG_PRINT //printf("%s():%d\n", __func__, __LINE__);
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
        //printf("get_value(): %d: %s rank %d, type %d\n", i, name.name(), data_rank, data_type);
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

//    int process_old(Xtc* xtc)
//    {
//        switch (xtc->contains.id()) {
//        case (TypeId::Parent): {
//            printf("Found TypeID == Parent, iterating...\n");
//            iterate(xtc);
//            break;
//        }
//        case (TypeId::Names): {
//            Names& names = *(Names*)xtc;
//            _namesLookup[names.namesId()] = NameIndex(names);
//            Alg& alg = names.alg();
//        printf("===============================\n");
//	    //printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
//        //           names.detName(), names.segment(), names.detType(),
//        //           alg.name(), alg.version(), (int)names.namesId());
//	    printf("process(): TypeId::Names: current token = %s\n", names.detName());
//	    //_CURRENT_PATH_TOKENS.push_back(names.detName());
//	    append_token(names.detName());
//        for (unsigned i = 0; i < names.num(); i++) {
//            Name& name = names.get(i);
//            printf("      process() level 2 token = %s\n", name.name());
//            //_CURRENT_PATH_TOKENS.push_back(name.name());
//            append_token(name.name());
//            print_local_path();
//            //printf("      Name: %s, Type: %d, Rank: %d.\n",name.name(),name.type(), name.rank());
//            pop_token();
//        }
//        pop_token();
//
//            break;
//        }
//        case (TypeId::ShapesData): {
//            ShapesData& shapesdata = *(ShapesData*)xtc;
//            // lookup the index of the names we are supposed to use
//            NamesId namesId = shapesdata.namesId();
//            // protect against the fact that this namesid
//            // may not have a NamesLookup.  cpo thinks this
//            // should be fatal, since it is a sign the xtc is "corrupted",
//            // in some sense.
//            if (_namesLookup.count(namesId)<=0) {
//                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
//                throw "invalid namesid";
//                break;
//            }
//            DescData descdata(shapesdata, _namesLookup[namesId]);
//            Names& names = descdata.nameindex().names();
//            Data& data = shapesdata.data();
//            printf("===============================\n");
//	    printf("ShapesData group:Found %d names, the token is detName = [%s]\n",names.num(), names.detName());
//	    append_token(names.detName());
//            for (unsigned i = 0; i < names.num(); i++) {
//                Name& name = names.get(i);
//                append_token(name.name());
//                print_local_path();
//                get_value(i, name, descdata);
//                pop_token();
//            }
//        pop_token();
//            break;
//        }
//        default:
//            break;
//        }
//        return Continue;
//    }

    int process(Xtc* xtc)
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

            printf("\n=============================== Found 2nd token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
            print_input_path();
            print_local_path();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("      process() level 2 token = %s\n", name.name());
                candidate_str = name.name();
                append_token(candidate_str);
                append_token(name.name());
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

            printf("\n=============================== Found 2nd token match: str = %s, index = %d\n", candidate_str.c_str(), _scan_index);
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
        _scan_index = 0;
    }
    int index_increment(){
        _scan_index++;
        return _scan_index;
    }
    int index_decrement(){
        _scan_index--;
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
        printf("current local path = ");
        _print_path(_CURRENT_PATH_TOKENS);
        printf("current input_path index = %d, segment = ", _scan_index);
        for(int i = 0; i <= _scan_index; i++){
            printf("/%s", _INPUT_PATH_TOKENS[i].c_str());
        }
        printf("\n");
    }
    int _scan_index; //mark which token to compare
    vector<string> _INPUT_PATH_TOKENS;
private:
    NamesLookup _namesLookup;
    vector<string> _CURRENT_PATH_TOKENS;


    void _print_path(vector<string>vec)
    {
        if(vec.size()==0)
            return;
        printf(" path = ");
        for(vector<string>::iterator it = vec.begin(); it != vec.end(); ++it){
            printf("/%s", (*it).c_str());
        }
        printf("\n");
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
    dbgiter._scan_index = -1;
    dbgiter.print_input_path();
    int i =0;

    dg = iter.next();//first dg, for configure transition.
    string candidate_str = string(TransitionId::name(dg->seq.service()));


//    printf("path string token = %s, input token = %s, scan_index = %d, comp = %d\n",
//            TransitionId::name(dg->seq.service()), dbgiter._INPUT_PATH_TOKENS[dbgiter._scan_index].c_str(), dbgiter._scan_index,
//            candidate_str.compare(dbgiter._INPUT_PATH_TOKENS[dbgiter._scan_index]));

    dbgiter.append_token(candidate_str);
    dbgiter.print_local_path();
    dbgiter.iterate(&(dg->xtc));
    dbgiter.pop_token();

    printf("\n=============================== Configure transition completed. ===============================\n");
    while ((dg = iter.next())) {//each data item in the file
        i++;
        nevent++;
//        printf("event %d, %s transition: time %d.%09d, pulseId 0x%lu, env 0x%lu, "
//               "payloadSize %d extent %d\n",
//               nevent,
//               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
//               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
//               dg->env, dg->xtc.sizeofPayload(),dg->xtc.extent);
        DEBUG_PRINT
        string candidate_str = string(TransitionId::name(dg->seq.service()));
        dbgiter.append_token(candidate_str);
        //dbgiter.index_increment();

//        printf("path string token = %s, input token = %s, scan_index = %d, comp = %d\n",
//                TransitionId::name(dg->seq.service()), dbgiter._INPUT_PATH_TOKENS[dbgiter._scan_index].c_str(), dbgiter._scan_index,
//                candidate_str.compare(dbgiter._INPUT_PATH_TOKENS[dbgiter._scan_index]));


        bool token_match = (candidate_str.compare(dbgiter._INPUT_PATH_TOKENS[dbgiter._scan_index]) == 0);



        if((candidate_str.compare(dbgiter._INPUT_PATH_TOKENS[dbgiter._scan_index]) == 0)){
            //printf("========= first section match, _scan_index = %d =========\n", dbgiter._scan_index);
            DEBUG_PRINT
            debugprint = true;
            //printf("\n=============================== Found token match: str = %s, index = %d\n", candidate_str.c_str(), dbgiter._scan_index);
            //dbgiter.print_input_path();
            //dbgiter.print_local_path();


            //if (debugprint) dbgiter.iterate(&(dg->xtc));
            //printf("===============================\n\n");

        }
        else{
            DEBUG_PRINT
            debugprint = false;
            //dbgiter.index_decrement();
            //dbgiter.pop_token();
            //continue;
        }
        //dbgiter.print_local_path();

        DEBUG_PRINT
        if (debugprint) {
            dbgiter.iterate(&(dg->xtc));
        }
        //if false, dg->xtc need to move to the next block.

        DEBUG_PRINT
        //dbgiter.index_decrement();
        dbgiter.pop_token();
    }

    //dbgiter.print_local_path();

    return it;
}


void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}




int main(int argc, char* argv[]){
    int c;
    char* xtcname = 0;
    int parseErr = 0;
    unsigned neventreq = 0xffffffff;
    bool debugprint = false;
    printf("XTC_IO_API test\n");
    while ((c = getopt(argc, argv, "hf:n:d")) != -1) {
        switch (c) {
        case 'h':
            usage(argv[0]);
            exit(0);
        case 'f':
            xtcname = optarg;
            break;
        case 'n':
            neventreq = atoi(optarg);
            break;
        case 'd':
            debugprint = true;
            break;
        default:
            parseErr++;
        }
    }

    if (!xtcname) {
        usage(argv[0]);
        exit(2);
    }

    int fd = open(xtcname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", xtcname);
        exit(2);
    }

    char* sample_path = "/L1Accept/xppcspad/arrayRaw";
    target_open(fd, sample_path);

    return 0;
}
//{
//    int c;
//    char* xtcname = 0;
//    int parseErr = 0;
//    unsigned neventreq = 0xffffffff;
//    bool debugprint = false;
//    printf("XTC_IO_API test\n");
//    while ((c = getopt(argc, argv, "hf:n:d")) != -1) {
//        switch (c) {
//        case 'h':
//            usage(argv[0]);
//            exit(0);
//        case 'f':
//            xtcname = optarg;
//            break;
//        case 'n':
//            neventreq = atoi(optarg);
//            break;
//        case 'd':
//            debugprint = true;
//            break;
//        default:
//            parseErr++;
//        }
//    }
//
//    if (!xtcname) {
//        usage(argv[0]);
//        exit(2);
//    }
//
//    int fd = open(xtcname, O_RDONLY);
//    if (fd < 0) {
//        fprintf(stderr, "Unable to open file '%s'\n", xtcname);
//        exit(2);
//    }
//
//    XtcFileIterator iter(fd, 0x4000000);
//    Dgram* dg;
//    unsigned nevent=0;
//    DebugIter dbgiter;
//    while ((dg = iter.next())) {//each data item in the file
//        if (nevent>=neventreq) break;
//        nevent++;
//        printf("event %d, %s transition: time %d.%09d, pulseId 0x%lu, env 0x%lu, "
//               "payloadSize %d extent %d\n",
//               nevent,
//               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
//               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
//               dg->env, dg->xtc.sizeofPayload(),dg->xtc.extent);
//
//        printf("path string token = %s\n", TransitionId::name(dg->seq.service()));
//        dbgiter.append_token(string(TransitionId::name(dg->seq.service())));
//        dbgiter.print_local_path();
//        if (debugprint) dbgiter.iterate(&(dg->xtc));
//        dbgiter.pop_token();
//    }
//    dbgiter.print_local_path();
//    ::close(fd);
//    return 0;
//}

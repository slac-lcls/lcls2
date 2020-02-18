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
#include <sys/time.h>
#include "xtc_io_api_c.h" //for extern C
#include "xtc_io_api_cpp.hh";
#include "xtc_tree.hh"

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

xtc_object* xtc_obj_new(int fd, void* fileIter, void* dbgiter, void* dg, const char* obj_path_abs);
xtc_object* xtc_obj_new(xtc_location* location, const char* obj_path_abs);

class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter() : XtcIterator()
    {
    }

    void get_ds_info(int i, Name& name, DescData& descdata){
        int array_size = name.rank();
        int data_type = name.type();
        name.name();//variable name
        name.str_type();//type name
    }

    void get_ds_data(int i, Name& name, DescData& descdata){

    }

    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();
        int data_type = name.type();
        printf("get_value() terminal data entry ==============  %d: %s rank %d, type %d\n", i, name.name(),
                data_rank, data_type);
        //printf("get_value() token = %s\n", name.name());
        switch(name.type()){
        case(Name::UINT8):{
            if(data_rank > 0){
                Array<uint8_t> arrT = descdata.get_array<uint8_t>(i);
                printf("type uint8_t: %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type uint8_t:  %s: %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(Name::UINT16):{
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(i);
                printf("type uint16_t:  %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type uint16_t: %s: %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(Name::UINT32):{
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(i);
                printf("type uint32_t: %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type uint32_t: %s: %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(Name::UINT64):{
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(i);
                printf("type uint64_t: %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type uint64_t %s: %d\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(Name::INT8):{
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(i);
                printf("type int8:  %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type int8: %s: %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(Name::INT16):{
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(i);
                printf("type int16: %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type int16: %s: %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(Name::INT32):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("type int32: %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type int32:   %s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::INT64):{
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(i);
                printf("type int64:  %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type int64:   %s: %d\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(Name::FLOAT):{
            if(data_rank > 0){
                Array<float> arrT = descdata.get_array<float>(i);
                printf("type float:  %s: %f, %f\n",name.name(),arrT.data()[0],arrT.data()[1]);
                    }
            else{
                printf("type float:  %s: %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(Name::DOUBLE):{
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(i);
                printf("type double: %s: %f, %f, %f\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type double: %s: %f\n",name.name(),descdata.get_value<double>(i));
            }
            break;
        }

        case(Name::CHARSTR):{
            if(data_rank > 0){
                Array<char> arrT = descdata.get_array<char>(i);
                printf("type charstr:  %s: \"%s\"\n",name.name(),arrT.data());
                    }
            else{
                printf("type charstr: %s: string with no rank?!?\n",name.name());
            }
            break;
        }

        case(Name::ENUMVAL):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("type ENUMVAL: %s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("type ENUMVAL %s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::ENUMDICT):{
            if(data_rank > 0){
                printf("type ENUMDICT %s: enumdict with rank?!?\n", name.name());
            } else{
                printf("type ENUMDICT  %s: %d\n",name.name(),descdata.get_value<int32_t>(i));
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
	    printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s, "
	            "Version: 0x%6.6x, namesid: 0x%x, Names:\n",
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
        printf("\nStarting new Xtc: xtc.extent = %lu\n", xtc->extent);
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
//      printf("===============================\n");
      printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s, "
              "Version: 0x%6.6x, namesid: 0x%x, Names:\n",
                   names.detName(), names.segment(), names.detType(),
                   alg.name(), alg.version(), (int)names.namesId());
      //printf("process(): TypeId::Names: current token = %s\n", names.detName());
      //_CURRENT_PATH_TOKENS.push_back(names.detName());
      string token = names.detName();
      token += "_";
      token += names.detType();
      token += "_";
      token += alg.name();
      append_token(token);
      append_token(to_string(names.segment()));
      xtc_object* new_obj = xtc_obj_new(CURRENT_LOCATION, get_current_path().c_str());
      xtc_tree_node_add(new_xtc_node(new_obj));
        for (unsigned i = 0; i < names.num(); i++) {
            Name& name = names.get(i);
//            if (strcmp(name.name(),"intOffset")==0) {
//                uint64_t offset    = descdata.get_value<uint64_t>(i);
//                printf("Type=Names  offset = 0x%x\n", offset);
//            }
            //printf("      process() level 2 token = %s\n", name.name());

            append_token(name.name());
            //print_local_path();
            string path_str = get_current_path();
            xtc_object* new_obj = xtc_obj_new(CURRENT_LOCATION, path_str.c_str());
            xtc_tree_node_add(new_xtc_node(new_obj));
            cout<<"Output path = " << path_str <<endl;
            //printf("      Name: %s, Type: %d, Rank: %d.\n",name.name(),name.type(), name.rank());
            pop_token();
        }
        pop_token();
        pop_token();

            break;
        }
        case (TypeId::ShapesData): {
            //printf("=============================== find ShapesData!!!\n");
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
      //printf("ShapesData group:Found %d names, the token is detName = [%s]\n",names.num(), names.detName());
            string token = names.detName();
            token += "_";
            token += names.detType();
            token += "_";
            token += names.alg().name();
            append_token(token);
            append_token(to_string(names.segment()));

            xtc_object* new_obj = xtc_obj_new(CURRENT_LOCATION, get_current_path().c_str());
            xtc_tree_node_add(new_xtc_node(new_obj));
      //append_token(names.detName());
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                if (strcmp(name.name(),"intOffset")==0) {
                    uint64_t offset    = descdata.get_value<uint64_t>(i);
                    printf("Type=ShapesData  offset = 0x%x\n", offset);
                }
                append_token(name.name());
                printf("\n");
                //print_local_path();
                string path_str = get_current_path();

                xtc_object* new_obj = xtc_obj_new(CURRENT_LOCATION, get_current_path().c_str());
                new_obj->obj_type = XTC_DS;
                xtc_tree_node_add(new_xtc_node(new_obj));

                cout<<"Output path = " << path_str <<endl;
                get_value(i, name, descdata);
                pop_token();
            }
        pop_token();
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
        //printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s,
            //Version: 0x%6.6x, namesid: 0x%x, Names:\n",
        //           names.detName(), names.segment(), names.detType(),
        //           alg.name(), alg.version(), (int)names.namesId());
        string candidate_str = names.detName();
        printf("process(): TypeId::Names: current token = [%s]\n", candidate_str.c_str());
        DEBUG_PRINT
        append_token(candidate_str);
        DEBUG_PRINT
        if(1){//always list config

            printf("\n=============================== Names:: Found 2nd token match: str = %s, index = %d\n",
                    candidate_str.c_str(), _scan_index);
            //print_input_path();
            //print_local_path();
//            for (unsigned i = 0; i < names.num(); i++) {
//                Name& name = names.get(i);
//                printf("      process() level 2 token = %s\n", name.name());
//                candidate_str = name.name();
//                append_token(candidate_str);
//                if(1){
//
//                    printf("\n=============================== Terminal token match: str = %s, index = %d\n",
//                            candidate_str.c_str(), _scan_index);
//                    print_input_path();
//                    print_local_path();
//                    printf("      Name: %s, Type: %d, Rank: %d.\n",name.name(),name.type(), name.rank());
//                    printf("===============================\n\n");
//
//                }
//                print_local_path();
//
//                pop_token();
//            }
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
//        if(get_index() <= SCAN_DEPTH){
//
//            printf("\n=============================== ShapesData:: Found 2nd token match: str = %s, index = %d\n",
//                    candidate_str.c_str(), _scan_index);
//            print_input_path();
//            print_local_path();
//            for (unsigned i = 0; i < names.num(); i++) {
//                Name& name = names.get(i);
//                candidate_str = name.name();
//                append_token(candidate_str);
//                print_local_path();
//
//                printf("      process() level 2 token = %s\n", name.name());
//                //index_increment();
//                if(get_index() <= SCAN_DEPTH){
//
//                    printf("\n=============================== Found terminal token match: str = %s, index = %d\n",
//                            candidate_str.c_str(), _scan_index);
//                    print_input_path();
//                    print_local_path();
//                    get_value(i, name, descdata); //this should be the terminal.
//                    printf("===============================\n\n");
//
//                }
//                //index_decrement();
//                pop_token();
//            }
//            printf("===============================\n\n");
//
//        }
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
            //printf("Found TypeID == Parent, iterating...\n");
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            DEBUG_PRINT
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();
        //printf("===============================\n");
        //printf("***  Per names metadata: DetName: %s, Segment %d, DetType: %s, Alg: %s,
        //Version: 0x%6.6x, namesid: 0x%x, Names:\n",
        //           names.detName(), names.segment(), names.detType(),
        //           alg.name(), alg.version(), (int)names.namesId());
//        string candidate_str = names.detName();
//        printf("process(): TypeId::Names: current token = [%s]\n", candidate_str.c_str());


        string candidate_str = names.detName();
        candidate_str += "_";
        candidate_str += names.detType();
        candidate_str += "_";
        candidate_str += names.alg().name();
        append_token(candidate_str);
        append_token(to_string(names.segment()));

        int match = compare_input_path();//1: continue to scan, -1 mismatch, 0 all match.

        if(match == 1){
            printf("\n=============================== Names:: Found 2nd token match: str = %s, index = %d\n",
                    candidate_str.c_str(), _scan_index);
            print_input_path();
            print_local_path();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("      process() level 2 token = %s\n", name.name());

                candidate_str = name.name();
                append_token(candidate_str);
                print_local_path();

                match = compare_input_path();
                if(match == 0){
                    printf("\n=============================== Found terminal token match: str = %s, index = %d\n",
                            candidate_str.c_str(), _scan_index);
                    print_input_path();
                    print_local_path();
                    printf("===============================\n\n");
                    //return 0;
                }
                pop_token();

                if(candidate_str.compare(_INPUT_PATH_TOKENS[_scan_index]) == 0){

                    printf("\n=============================== Terminal token match: str = %s, index = %d\n",
                            candidate_str.c_str(), _scan_index);
                    print_input_path();
                    print_local_path();
                    //return 0;
                    printf("      Name: %s, Type: %d, Rank: %d.\n",name.name(),name.type(), name.rank());
                    printf("===============================\n\n");

                }

            }
            //printf("===============================\n\n");

        } else if(match == 0 ) {//all match.
            printf("Find target group: ");
            print_local_path();
            //need to return something?
        }
        else if(match == 2) {
            printf("Scanned too much: ");
            print_local_path();
        }
//        index_decrement();

        pop_token();
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
            candidate_str += "_";
            candidate_str += names.detType();
            candidate_str += "_";
            candidate_str += names.alg().name();
            append_token(candidate_str);
            append_token(to_string(names.segment()));

            int match = compare_input_path();//1: continue to scan, -1 mismatch, 0 all match.
            //printf("Compare: canditate_str = %s, input_path_token[%d] = %s, match = %d\n",
            //candidate_str.c_str(), _scan_index , _INPUT_PATH_TOKENS[_scan_index].c_str(), match);
            if(match == 1 ){
                //candidate_str.compare(_INPUT_PATH_TOKENS[_scan_index]) == 0
                printf("\n=============================== ShapesData:: Found 2nd token match: str = %s, index = %d\n",
                        candidate_str.c_str(), _scan_index);
                print_input_path();
                print_local_path();
                for (unsigned i = 0; i < names.num(); i++) {
                    Name& name = names.get(i);
                    candidate_str = name.name();
                    append_token(candidate_str);
                    print_local_path();
                    match = compare_input_path();
                    if(match == 0){
                        printf("\n=============================== Found terminal token match: str = %s, index = %d\n",
                                candidate_str.c_str(), _scan_index);
                        print_input_path();
                        print_local_path();
                        get_value(i, name, descdata); //this should be the terminal.
                        printf("===============================\n\n");
                    }
                    pop_token();
                }
                printf("===============================\n\n");
            }
            else if(match == 0 ) {//all match.
                printf("Find target group: ");
                print_local_path();
            }
            else if(match == 2) {
                printf("Scanned too much: ");
                print_local_path();
            }
        pop_token();//pop segment #
        pop_token();// pop detName, alg, etc,.
            break;
        }// end case ShapesData
        default:
            break;
    }//end switch
        return Continue;
    }
    void index_init(){
        _scan_index = -1;
    }

    int get_index(){
        return _scan_index;
    }

    void append_token(string str){
        _CURRENT_PATH_TOKENS.push_back(str);
        _index_increment();
        //cout << "After append: input token:"<< _INPUT_PATH_TOKENS[_scan_index] << ", index = "<< _scan_index <<endl;
        //index_print();
    }
    void set_input_path(vector<string> str_vec){
        _INPUT_PATH_TOKENS = str_vec;
    }

    int compare_input_path(){
        //_INPUT_PATH_TOKENS;
        //_CURRENT_PATH_TOKENS;
        for(int i = 0; i < min(_INPUT_PATH_TOKENS.size(), _CURRENT_PATH_TOKENS.size()); i++){
            if(_INPUT_PATH_TOKENS[i].compare(_CURRENT_PATH_TOKENS[i]) != 0)
                return -1;
        }
        if(_CURRENT_PATH_TOKENS.size() < _INPUT_PATH_TOKENS.size())
            return 1;//continue to scan
        if(_CURRENT_PATH_TOKENS.size() == _INPUT_PATH_TOKENS.size())
            return 0;//find a perfect match
        if(_CURRENT_PATH_TOKENS.size() > _INPUT_PATH_TOKENS.size())
            return 2;//scan too much, should return one step earlier.
        return 0;
    }
    void pop_token(){
        _CURRENT_PATH_TOKENS.pop_back();
        _index_decrement();
        //cout << "After pop: input token:"<< _INPUT_PATH_TOKENS[_scan_index] << ", index = "<< _scan_index <<endl;
        //index_print();
    }
    void print_input_path(){
        printf("input path = ");
        print_path(_INPUT_PATH_TOKENS);
    }
    void print_local_path(void){
        printf("current local");
        print_path(_CURRENT_PATH_TOKENS);
//        printf("current input_path index = %d, segment = ", _scan_index);
//        for(int i = 0; i <= _scan_index; i++){
//            printf("/%s", _INPUT_PATH_TOKENS[i].c_str());
//        }
        printf("\n");
    }

    string get_current_path(){
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

    void xtc_tree_init(xtc_object* root_obj){
        ROOT_NODE = new_xtc_node(root_obj);
    }
    void xtc_tree_init(int fd){
        xtc_object* root_obj = (xtc_object*)calloc(1, sizeof(xtc_object));
        root_obj->fd = fd;
        root_obj->obj_path_abs = strdup("/");
        xtc_token_new((xtc_token_t*)(root_obj->obj_token), 16);
        ROOT_NODE = new_xtc_node(root_obj);
    }

    int xtc_tree_node_add(xtc_node* node){
        return add_xtc_node(ROOT_NODE, node);
    }

    int xtc_tree_print(){
        return print_tree(ROOT_NODE);
    }
    int attach_helper(xtc_object* h){
        assert(h);
        h->ref_cnt++;
        return 0;
    }

    int detach_helper(){
        extern_helper->ref_cnt--;
        return 0;
    }

    xtc_object* get_helper(){
        return extern_helper;
    }

    xtc_location* CURRENT_LOCATION;

    vector<string> _CURRENT_PATH_TOKENS;


private:
    ItType iterator_type;
    NamesLookup _namesLookup;

    xtc_object* extern_helper;
    xtc_node* ROOT_NODE;

    int _index_increment(){
        _scan_index++;
        //cout<<"after increment: index = "<<_scan_index<<endl;
        return _scan_index;
    }
    int _index_decrement(){
        _scan_index--;
        //cout<<"after decrement: index = "<<_scan_index<<endl;
        return _scan_index;
    }
};

void print_path(vector<string>vec)
{
    if(vec.size()==0)
        return;
    printf(" path = ");
    for(vector<string>::iterator it = vec.begin(); it != vec.end(); ++it){
        printf("/%s", (*it).c_str());
    }
    //printf("\n");
}

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

string tok_to_str(vector<string> token_list){
    string s("/");
    for(auto i : token_list){
        s += i;
        if(i != token_list.back())
            s += "/";
    }
    return s;
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

    //location init
    dbgiter.CURRENT_LOCATION = (xtc_location*)calloc(1, sizeof(xtc_location));
    dbgiter.CURRENT_LOCATION->fd = fd;
    dbgiter.CURRENT_LOCATION->fileIter = (void*)(&iter);
    dbgiter.CURRENT_LOCATION->dbgiter = (void*)(&dbgiter);
    dbgiter.CURRENT_LOCATION->dg = NULL;

    dbgiter.index_init();//-1
    dbgiter.set_iterator_type(DebugIter::LIST_ALL);
    int i = 0;

    dg = iter.next();//first dg, for configure transition.
    dbgiter.CURRENT_LOCATION->dg = dg;
    //int fd, XtcFileIterator* fileIter, DebugIter* dbgiter, Dgram* dg
    xtc_object* head_obj = xtc_obj_new(fd, &iter, &dbgiter, dg, "/");

    head_obj->obj_type = XTC_ROOT_GROUP;
    dbgiter.xtc_tree_init(head_obj);

    string candidate_str = string(TransitionId::name(dg->service()));
    //candidate_str += "_";
    //candidate_str += to_string(dg->time.asDouble());
    dbgiter.append_token(candidate_str);
    dbgiter.print_local_path();

    //
    dbgiter.iterate(&(dg->xtc));
    dbgiter.pop_token();
    int L1accept_cnt = 0;
    printf("\n=============================== Configure transition completed. ===============================\n");
    while ((dg = iter.next())) {//each data item in the file
        dbgiter.CURRENT_LOCATION->dg = dg;
        i++;
        nevent++;

        DEBUG_PRINT
        string candidate_str = string(TransitionId::name(dg->service()));
        candidate_str += "/";
        candidate_str += to_string(dg->time.asDouble());
//        if(candidate_str.compare("L1Accept") == 0){
//            L1accept_cnt++;
//            candidate_str += "_";
//            candidate_str += to_string(L1accept_cnt);
//        }

        dbgiter.append_token(candidate_str);
        dbgiter.print_local_path();

        xtc_object* xtc_obj = xtc_obj_new(fd, &iter, &dbgiter, dg, dbgiter.get_current_path().c_str());
        dbgiter.xtc_tree_node_add(new_xtc_node(xtc_obj));

        DEBUG_PRINT
        if (debugprint) {
            dbgiter.iterate(&(dg->xtc));
        }
        //if false, dg->xtc need to move to the next block.

        DEBUG_PRINT
        dbgiter.pop_token();
    }
    printf("\n\n\n\n");
    dbgiter.xtc_tree_print();

    return it;

}



xtc_object* xtc_obj_new(int fd, void* fileIter, void* dbgiter, void* dg, const char* obj_path_abs){
    xtc_object* ret = (xtc_object*)calloc(1, sizeof(xtc_object));
    ret->fd = fd;

    ret->location = (xtc_location*) calloc(1, sizeof(xtc_location));
    ret->location->fd = fd;
    ret->location->dg = dg;
    ret->location->dbgiter = dbgiter;
    ret->location->fileIter = fileIter;

    ret->obj_type = XTC_GROUP;
    ret->obj_token = calloc(1, sizeof(xtc_token_t));
    ret->obj_path_abs = strdup(obj_path_abs);
    xtc_token_new((xtc_token_t*)ret->obj_token, 16);
    return ret;
}

xtc_object* xtc_obj_new(xtc_location* location, const char* obj_path_abs){
    assert(location);
    xtc_object* obj = xtc_obj_new(location->fd,
            location->fileIter, location->dbgiter, location->dg, obj_path_abs);
    return obj;
}
xtc_object* _target_open(xtc_object* helper_in, const char* obj_vol_path){

    assert(helper_in && obj_vol_path);

    //int new_fd = open("data.xtc2", O_RDONLY);
    int fd = helper_in->fd;
    //cout<<"received fd = "<< fd <<", new_fd = "<<new_fd<<endl;

    //XtcFileIterator iter(fd, 0x4000000);
    if(0 == strcmp(obj_vol_path, "/")){
        return helper_in;
    }

    XtcFileIterator* iter = (XtcFileIterator*)(helper_in->location->fileIter);
    Dgram* dg;
    unsigned nevent=0;

    DebugIter* dbgiter = (DebugIter*)(helper_in->location->dbgiter);//new DebugIter();
    bool debugprint = true;
    DEBUG_PRINT
    dbgiter->set_input_path(str_tok(obj_vol_path, "/ "));
    dbgiter->print_input_path();
    dbgiter->index_init();//set to -1;
    dbgiter->set_iterator_type(DebugIter::SEARCH);
    dbgiter->print_input_path();
//    int i =0;
//    DEBUG_PRINT
//    dg = iter->next();//first dg, for configure transition.
//    string candidate_str = string(TransitionId::name(dg->service()));
//    DEBUG_PRINT
//    dbgiter->append_token(candidate_str);
//    dbgiter->print_local_path();
//    DEBUG_PRINT
//    dbgiter->iterate(&(dg->xtc));
//    DEBUG_PRINT
//    dbgiter->pop_token();

    printf("\n=============================== Configure transition completed. ===============================\n");
    while ((dg = iter->next())) {//each data item in the file
        DEBUG_PRINT
        string candidate_str = string(TransitionId::name(dg->service()));
        to_string(dg->time.asDouble());
        dbgiter->append_token(candidate_str);
        bool token_match = (candidate_str.compare(dbgiter->_INPUT_PATH_TOKENS[0]) == 0);//dbgiter->_scan_index
        if(token_match){
            printf("========= Transition match (%s), _scan_index = %d =========\n", candidate_str.c_str(), dbgiter->_scan_index);
            DEBUG_PRINT
            debugprint = true;
        }
        else{//skip
            DEBUG_PRINT
            printf("========= Transition doesn't match (%s), skip =========\n", candidate_str.c_str());
            debugprint = false;
        }

        DEBUG_PRINT
        if (debugprint) {
            dbgiter->iterate(&(dg->xtc));
        }
        //if false, dg->xtc need to move to the next block.
        DEBUG_PRINT

        dbgiter->pop_token();
    }
    xtc_object* ret = (xtc_object*)calloc(1, sizeof(xtc_object));
    ret->fd = fd;
    ret->location->dg = (void*)dg;
    ret->location->dbgiter = (void*)dbgiter;
    return ret;
}


unsigned long xtc_token_new(xtc_token_t* token, unsigned int h5_token_size) {
    if(!token)
        token = (xtc_token_t*)calloc(1, sizeof(xtc_token_t));

    struct timeval tp;
    gettimeofday(&tp, NULL);
    int32_t tval = (int32_t)((1000000 * tp.tv_sec) + tp.tv_usec);
    int n = sizeof(int32_t)/sizeof(uint8_t);
    for(int i = 0; i < h5_token_size; i++){//16
        if(i < n)
            token->__data[i] = *(uint8_t*)(&tval + i * sizeof(uint8_t));
        else
            token->__data[i] = 0;
    }
    return tval;
}


xtc_object* xtc_helper_new(){
    xtc_object* ret = (xtc_object*)calloc(1, sizeof(xtc_object));
    ret->obj_token = calloc(1, sizeof(xtc_token_t));
    return ret;
}

xtc_object* _file_open(int fd){
    XtcFileIterator* fileIter = new XtcFileIterator(fd, 0x4000000);
    DebugIter* dbgiter = new DebugIter();
    bool debugprint = true;
    DEBUG_PRINT
    dbgiter->index_init();//-1
    dbgiter->set_iterator_type(DebugIter::LIST_ALL);//LIST_W_DEPTH
    dbgiter->SCAN_DEPTH = 1;
    int i = 0;
    DEBUG_PRINT
    Dgram* dg = new Dgram();
    DEBUG_PRINT
    dg = fileIter->next();//first dg, for configure transition.
    DEBUG_PRINT
    string candidate_str = string(TransitionId::name(dg->service()));
    DEBUG_PRINT
    dbgiter->append_token(candidate_str);
    //dbgiter->print_local_path();
    DEBUG_PRINT
    dbgiter->iterate(&(dg->xtc));
    dbgiter->pop_token();
    DEBUG_PRINT
    xtc_object* head_obj = xtc_obj_new(fd, fileIter, dbgiter, dg, "/");//????  start after initial scan for configure transitions

    printf("\n=========================== Configure transition completed. ===========================\n");
    DEBUG_PRINT
    return head_obj;

}



EXTERNC xtc_object* xtc_file_open(char* file_path){
    printf("xtc_file_open() is called\n");
    int fd = open(file_path, O_RDONLY);
    xtc_object* ret = _file_open(fd);//finished config reading
    ret->ref_cnt = 0;// leave ref_cnt to VOL.
    return ret;
}

//
EXTERNC xtc_object* xtc_path_search(xtc_object* file, char* path){
    assert(file && file->location->dbgiter);
    cout <<"Searching path: "<< path <<endl;
    xtc_object* path_obj = _target_open(file, path);
    path_obj->ref_cnt = 0;
    return path_obj;
}
//open it and what's next?
EXTERNC xtc_func_t xtc_it_open(void* param){
    xtc_object* p = (xtc_object*)param;
    Dgram* dg = (Dgram*) p->location->dg;
    XtcFileIterator iter(p->fd, 0x4000000);
    DebugIter* dbgiter = (DebugIter*) p->location->dbgiter;

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

EXTERNC void xtc_file_close(xtc_object* helper){
    assert(helper->ref_cnt == 0);
    close(helper->fd);
    //delete helper->dbgiter;
    //delete (Dgram*)(helper->target_it);
    free(helper);
}
void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}


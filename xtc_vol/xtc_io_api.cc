/** \file xtc_io_api.cc
 * @brief XTC2 iterator and utility functions, used to scan xtc2 file and smd file to load the metadata and to build the index tree.
 *
 */
#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <unistd.h>

#include <unordered_map>

#include <xtcdata/xtc/XtcFileIterator.hh>
#include <xtcdata/xtc/XtcIterator.hh>
#include <xtcdata/xtc/ShapesData.hh>
#include <xtcdata/xtc/DescData.hh>

#include <vector>
#include <sys/time.h>
#include "xtc_io_api_c.h" //for extern C
#include "xtc_io_api_cpp.hh"
#include "xtc_tree.hh"

using namespace XtcData;
using namespace std;
using std::string;


#define TIMESTAMP_DS_NAME "timestamps"
#define DEBUG_PRINT printf("%s():%d\n", __func__, __LINE__);
//H5 utility function prototype
//H5 name to xtc name, second
char* name_convert(const char* h5_name){
    char* xtc_name_surfix;

    return xtc_name_surfix;
}

int check_update_dgram(int fd, xtc_dgram_info* dg_info, Dgram* current_dgram_in_out);

/**
 * SMD iterator, used to read smd file that stores metadata and transition/dgram offset information of the original xtc2 file.
 */
class SmdIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    SmdIter() : XtcIterator()
    {
    }

    int process(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            _namesLookup[names.namesId()] = NameIndex(names);
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
            // see if the offset is in this ShapesData xtc
            Names& names = descdata.nameindex().names();
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                if (strcmp(name.name(),"intOffset")==0) {
                    offset = descdata.get_value<uint64_t>(i);
                }
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    void reset() {offset=-1;}
    int64_t offset;

private:
    NamesLookup _namesLookup;
};

xtc_object* xtc_obj_new(int fd, void* fileIter, void* dbgiter, void* dg, const char* obj_path_abs);
xtc_object* xtc_obj_new(xtc_location* location, const char* obj_path_abs);

/**
 * Customized XTC iterator, it's the core part to read xtc2 file.
 * It read the file, and extract hierarchy information and store in a index tree.
 */
class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter() : XtcIterator()
    {

    }
    /**
     * Collect dataset information and build structures for HDF5 VOL to build data space,
     * also includes the position info in the xtc2 file.
     */
    xtc_ds_info* get_ds_info(int index, ShapesData* shapesdata, xtc_dgram_info &dgram_info){//int i Names& names,
        NamesId namesId = shapesdata->namesId();

        DescData descdata(*shapesdata, _namesLookup[namesId]);
        DEBUG_PRINT
        Names& names = descdata.nameindex().names();

        xtc_ds_info* dataset_info = (xtc_ds_info*)calloc(1, sizeof(xtc_ds_info));
        Name& name = names.get(index);

        printf("get_ds_info: index = %d, shapesdata = %p, &shapesdata = %p, offset = %d, dgram_info.isL1 = %d, dgram = %p, dgram_id = 0x%llx, sizeofPayload = %d, field_name = [%s], str_type = [%s]\n",
                index, shapesdata, &shapesdata, dgram_info.shapesdata_offset, dgram_info.isL1,
                _current_dgram, _current_dgram->time.value(), _current_dgram->xtc.sizeofPayload(),
                name.name(), name.str_type());

        dataset_info->type = (xtc_data_type)(int)(name.type());
        printf("get_ds_info: set data type: %d\n", dataset_info->type);
        dataset_info->dim_cnt = name.rank();//# of dimension: 0-5, how many dimensions of the shape:

        dataset_info->element_cnt = name.get_element_size(name.type());// how many elements

        dataset_info->maximum_dims = NULL;

        uint32_t* shape = descdata.shape(name);

        dataset_info->total_pixel_cnt = 1;

        if(name.type() == Name::CHARSTR){//single string
            dataset_info->dim_cnt = 1;
            dataset_info->current_dims[0] = 1;
        } else {
            for(int i = 0; i < name.rank(); i++)
                dataset_info->current_dims[i] = shape[i];

            if(dataset_info->dim_cnt > 0){
                printf("dim_cnt = %d, dimensions val: ", dataset_info->dim_cnt);
                for(int i = 0; i < dataset_info->dim_cnt; i++){
                    dataset_info->total_pixel_cnt *= dataset_info->current_dims[i];
                    printf("%d, ", dataset_info->current_dims[i]);
                }
                printf("\n");
            }//dim_cnt == 0
        }
        if(dgram_info.ds_type < 0 || dgram_info.ds_type > 2){
            assert(0);
        }
        dataset_info->ds_type = dgram_info.ds_type;
        dataset_info->data_handle = (xtc_data_handle*)calloc(1, sizeof(xtc_data_handle));
        dataset_info->data_handle->dgram_id = dgram_info.dgram_id;

        dataset_info->data_handle->index = index;

        dataset_info->data_handle->dgram_info = (xtc_dgram_info*)calloc(1, sizeof(xtc_dgram_info));
        dataset_info->data_handle->dgram_info->dgram_id = dgram_info.dgram_id;
        dataset_info->data_handle->dgram_info->dgram_offset = dgram_info.dgram_offset;
        dataset_info->data_handle->dgram_info->shapesdata_offset = dgram_info.shapesdata_offset;
        dataset_info->data_handle->dgram_info->isL1 = dgram_info.isL1;
        dataset_info->data_handle->dgram_info->nonL1Dgram = dgram_info.nonL1Dgram;
        dataset_info->data_handle->dgram_info->ds_type = dgram_info.ds_type;

        DEBUG_PRINT
        dataset_info->data_handle->shapesData = calloc(1, sizeof(void*));
        memcpy(dataset_info->data_handle->shapesData, &shapesdata, sizeof(void*));
        DEBUG_PRINT

        DEBUG_PRINT
        printf("get_ds_info: index = %d, copied shapesdata address = %p\n", index, shapesdata);

        dataset_info->isTimestampsDS = 0;
        printf("get_ds_infodata_handle: xtc_ptr = %p\n",
                dataset_info->data_handle->xtc_ptr);

        return dataset_info;
    }

    /**
     * Get data from nested and recursive xtc data structure and give it to dataset_read_all().
     * Only work with valid current Dgram, need to check the type before calling this, must be non-timestamp dataset.
     */
    size_t get_data(xtc_object* xtc_obj, size_t pixel_cnt, void* data_out){//Names* namesd,
        xtc_data_handle* data_handle = xtc_obj->ds_info->data_handle;
        assert(data_handle);
        DEBUG_PRINT
        Dgram* sys_current_dgram = (Dgram*)xtc_obj->location->dg;
        printf("get_data: this = %p, sys_current_dgram = %p, \n", this, sys_current_dgram);
        assert(sys_current_dgram);

        if(data_handle->dgram_id != sys_current_dgram->time.value()){//different dgram, need to load.
            auto it =_index_map->find(data_handle->dgram_id);
            if(it == _index_map->end()){
                printf("Can not find dgram info, id/timestamp = 0x%llx\n", data_handle->dgram_id);
                return -1;
            } else {
                printf(" Before: Current dgram id = 0x%llx\n", sys_current_dgram->time.value());
                check_update_dgram(this->get_fd(), &(it->second), sys_current_dgram);
                printf(" After: Current dgram id = 0x%llx\n", sys_current_dgram->time.value());
            }
        }

        ShapesData* shapesdata = (ShapesData*)((char*)sys_current_dgram + data_handle->dgram_info->shapesdata_offset);
        int index = data_handle->index;
        NamesId namesId = shapesdata->namesId();
        printf("namesId._value = 0x%lx\n", namesId.value());
        DescData descdata = DescData(*shapesdata, _namesLookup[namesId]);
        Names& names = descdata.nameindex().names();
        Name& name = names.get(index);

        assert(shapesdata);
        size_t pixel_size_out = 0;

        int data_rank = name.rank();
        int data_type = name.type();
        void* ret_data = NULL;
        char* tmp_buf = NULL;

        switch(name.type()){
        case(Name::UINT8):{
            assert(name.type() == xtc_data_type::UINT8);
            if(data_rank > 0){
                Array<uint8_t> arrT = descdata.get_array<uint8_t>(index);
                auto t = arrT.shape();
                //rank == 3 5*6*7 is to describe the shape dimension of a element
                //t[0] = 5, t[1]=6, t[2]=7

                ret_data = arrT.data();
                    }
            else{
                uint8_t val = descdata.get_value<uint8_t>(index);
                ret_data = &val;
            }
            pixel_size_out = sizeof(uint8_t);
            break;
        }

        case(Name::UINT16):{
            assert(name.type() == xtc_data_type::UINT16);
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(index);
                ret_data =arrT.data();
                    }
            else{
                uint16_t val = descdata.get_value<uint16_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(uint16_t);
            break;
        }

        case(Name::UINT32):{
            assert(name.type() == xtc_data_type::UINT32);
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(index);
                ret_data =arrT.data();
                    }
            else{
                uint32_t val = descdata.get_value<uint32_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(uint32_t);
            break;
        }

        case(Name::UINT64):{
            assert(name.type() == xtc_data_type::UINT64);
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(index);
                ret_data =arrT.data();
                    }
            else{
                uint64_t val = descdata.get_value<uint64_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(uint64_t);
            break;
        }

        case(Name::INT8):{
            assert(name.type() == xtc_data_type::INT8);
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(index);
                ret_data =arrT.data();
                    }
            else{
                int8_t val = descdata.get_value<int8_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(int8_t);
            break;
        }

        case(Name::INT16):{
            assert(name.type() == xtc_data_type::INT16);
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(index);
                ret_data =arrT.data();
                    }
            else{
                int16_t val = descdata.get_value<int16_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(int16_t);
            break;
        }

        case(Name::INT32):{
            assert(name.type() == xtc_data_type::INT32);
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(index);
                ret_data =arrT.data();
                    }
            else{
                int32_t val = descdata.get_value<int32_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(int32_t);
            break;
        }

        case(Name::INT64):{
            assert(name.type() == xtc_data_type::INT64);
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(index);
                ret_data =arrT.data();
                    }
            else{
                int64_t val = descdata.get_value<int64_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(int64_t);
            break;
        }

        case(Name::FLOAT):{
            assert(name.type() == xtc_data_type::FLOAT);
            if(data_rank > 0){
                DEBUG_PRINT
                Array<float> arrT = descdata.get_array<float>(index);
                DEBUG_PRINT
                ret_data =arrT.data();
                DEBUG_PRINT
                    }
            else{
                float val = descdata.get_value<float>(index);
                DEBUG_PRINT
                ret_data =&val;
                DEBUG_PRINT
            }
            DEBUG_PRINT
            pixel_size_out = sizeof(float);
            DEBUG_PRINT
            break;
        }

        case(Name::DOUBLE):{
            assert(name.type() == xtc_data_type::DOUBLE);
            DEBUG_PRINT
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(index);
                ret_data =arrT.data();
                    }
            else{
                double val = descdata.get_value<double>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(double);
            break;
        }

        case(Name::CHARSTR):{
            assert(name.type() == xtc_data_type::CHARSTR);
            assert(0);
            if(data_rank > 0){
                Array<char> arrT = descdata.get_array<char>(index);
                tmp_buf = (char*)calloc(arrT.num_elem() + 1, sizeof(char));

                for(int i = 0; i < arrT.num_elem(); i++){
                    tmp_buf[i] = arrT.data()[i];
                }
                assert(0);
                ret_data = &tmp_buf;
            }
            else{
                char val = descdata.get_value<char>(index);
                ret_data =&val;
            }
            DEBUG_PRINT
            assert(0);
            pixel_size_out = sizeof(char*);
            break;
        }

        case(Name::ENUMVAL):{
            assert(name.type() == xtc_data_type::ENUMVAL);
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(index);
                ret_data =arrT.data();
                    }
            else{
                int32_t val = descdata.get_value<int32_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(int32_t);
            break;
        }

        case(Name::ENUMDICT):{
            assert(name.type() == xtc_data_type::ENUMDICT);
            if(data_rank > 0){
                ret_data =NULL;
            } else{
                int32_t val = descdata.get_value<int32_t>(index);
                ret_data =&val;
            }
            pixel_size_out = sizeof(int32_t);
            break;
        }
        default:
            assert(0 && "Unsupported type.");
            break;
        }

        size_t read_size = pixel_cnt * pixel_size_out;
        memcpy(data_out, ret_data, read_size);
        return read_size;
    }

    /**
     * Print data value, used for demo and debugging.
     */
    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();
        int data_type = name.type();
        printf("get_value() terminal data entry ==============  %d: name:[%s], rank = %d, type = %d\n",
                i, name.name(), data_rank, data_type);
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
                printf("type charstr:  rank = %d, num_elem = %lu, name = [%s], str = [%s], data() = %p \n",
                        arrT.rank(), arrT.num_elem(), name.name(), arrT.data(), arrT.data());
                    }
            else{
                printf("type charstr: %s: string with 0 rank?!?\n",name.name());
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

    /**
     * This function is an implementation of Xtc::iterate() interface.
     */
    int process(Xtc* xtc){
        int ret = -1;;
        switch(get_iterator_type()){
            case LIST_ALL:
                ret = process_list_all(xtc);
                break;
            default:
                ret = -1;
                break;
        }
        return ret;
    }

    /**
     * Currently the only scan type is to scan all to build the hierarchy.
     */
    int process_list_all(Xtc* xtc)
    {
        assert(this->_current_dgram);

        uint64_t dgram_id = this->_current_dgram->time.value();
        xtc_dgram_info dgram_info;
        printf("%s:%d:  dbgiter = %p, this->_current_dgram = %p, dgram_id = 0x%llx\n",
                __func__, __LINE__, this, this->_current_dgram, dgram_id);
        //Ensure the smd data is valid for the xtc2 file: all things here should have an entry in the index.
        if(find_index(dgram_id, &dgram_info) != 0)
            return -1;

        printf("dgram_id = 0x%llx, dgram_info.isL1 = %d, ds_type = %d\n", dgram_info.dgram_id, dgram_info.isL1, dgram_info.ds_type);
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            printf("Found TypeID == Parent, iterating...\n");
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            printf("=============================== find Names!!!\n");
            Names& names = *(Names*)xtc;
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();

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
                append_token(name.name());
                //print_local_path();
                string path_str = get_current_path();
                xtc_object* new_obj = xtc_obj_new(CURRENT_LOCATION, path_str.c_str());
                xtc_tree_node_add(new_xtc_node(new_obj));
                pop_token();
            }
            pop_token();//segment
            pop_token();//detName_detType_alg
            break;
        }
        case (TypeId::ShapesData): {
            printf("=============================== find ShapesData!!!\n");

            int shapesdata_offset = (char*)xtc - (char*)(this->_current_dgram);


            ShapesData* shapesdata = (ShapesData*)calloc(1, sizeof(ShapesData*));
            shapesdata = static_cast<ShapesData*>(xtc);//if sure it is such a subclass type

            NamesId namesId = shapesdata->namesId();

            if (_namesLookup.count(namesId)<=0) {
                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
                throw "invalid namesid";
                break;
            }
            DescData descdata(*shapesdata, _namesLookup[namesId]);

            Names& names = descdata.nameindex().names();
            Data& data = shapesdata->data();

            string token = names.detName();
            token += "_";
            token += names.detType();
            token += "_";
            token += names.alg().name();
            append_token(token);
            append_token(to_string(names.segment()));

            string time_stamp_str = to_string(((Dgram*)CURRENT_LOCATION->dg)->time.asDouble());

            //check DS holder group existence.
            string cur_path = get_current_path();
            printf("search cur_path = %s\n", cur_path.c_str());
            xtc_node* group_ds_holder = xtc_tree_node_find(&cur_path); //the direct holder group of datasets
            if(!group_ds_holder){
                printf("Create group_ds_holder: %s\n", cur_path.c_str());
                xtc_object* new_obj = xtc_obj_new(CURRENT_LOCATION, get_current_path().c_str());
                xtc_tree_node_add(new_xtc_node(new_obj));
            } //do nothing if group exists.

            //  Store time stamps in a virtual dataset
            cur_path += "/";
            cur_path += TIMESTAMP_DS_NAME;
            xtc_node* timestamp_ds_node = xtc_tree_node_find(&cur_path);
            double time_stamp = ((Dgram*)CURRENT_LOCATION->dg)->time.asDouble();

            if(!timestamp_ds_node){// create timestamp dataset
                printf("Create timestamp_ds: %s\n", cur_path.c_str());
                append_token(TIMESTAMP_DS_NAME);
                xtc_object* new_obj = xtc_obj_new(CURRENT_LOCATION, get_current_path().c_str());
                new_obj->obj_type = XTC_TIME_DS;//XTC_TIME_DS
                vector<double>* tsv = new vector<double>();
                tsv->push_back(time_stamp);
                new_obj->data = (void*) tsv;
                xtc_ds_info* ds_info = (xtc_ds_info*)calloc(1, sizeof(xtc_ds_info));
                ds_info->current_dims[0] = 1;
                ds_info->type = DOUBLE;
                ds_info->element_cnt = 1;
                ds_info->dim_cnt = 1; //1D array
                ds_info->maximum_dims = NULL;
                ds_info->isTimestampsDS = 1;
                ds_info->ds_type = DS_TIMESTAMP;
                new_obj->ds_info = ds_info;
                new_obj->ds_info->data_handle = NULL;
                xtc_tree_node_add(new_xtc_node(new_obj));
                pop_token();//ds name
            } else {//add timestamp to dataset
                assert(timestamp_ds_node->my_obj);
                assert(timestamp_ds_node->my_obj->data);
                ((vector<double>*)(timestamp_ds_node->my_obj->data))->push_back(time_stamp);
                timestamp_ds_node->my_obj->ds_info->element_cnt =
                        ((vector<double>*)(timestamp_ds_node->my_obj->data))->size();
            }

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                if (strcmp(name.name(),"intOffset")==0) {
                    uint64_t offset = descdata.get_value<uint64_t>(i);
                    //printf("Type=ShapesData  offset = 0x%x\n", offset);
                }
                append_token(name.name());//ds name
                print_local_path();
                string path_str = get_current_path();
                xtc_object* new_obj = xtc_obj_new(CURRENT_LOCATION, get_current_path().c_str());
                new_obj->obj_type = XTC_DS;

                //For both L1 and non-L1 datasets
                dgram_info.shapesdata_offset = shapesdata_offset;
                new_obj->ds_info = get_ds_info(i, shapesdata, dgram_info);
                xtc_tree_node_add(new_xtc_node(new_obj));
                get_value(i, name, descdata);
                pop_token();//ds name
            }

        pop_token();//segment
        pop_token();//detName_detType_alg
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

    int get_index(){
        return _scan_index;
    }

    void append_token(string str){
        _CURRENT_PATH_TOKENS.push_back(str);
        _index_increment();
    }
    void set_input_path(vector<string> str_vec){
        _INPUT_PATH_TOKENS = str_vec;
    }

    int compare_input_path(){
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
    }
    void print_input_path(){
        printf("input path = ");
        print_path(_INPUT_PATH_TOKENS);
    }
    void print_local_path(void){
        printf("current local");
        print_path(_CURRENT_PATH_TOKENS);
        printf("\n");
    }

    string get_current_path(){
        string ret("/");
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

    vector<string> _INPUT_PATH_TOKENS;

    /**
     * The tree-related functions below are used to build and access the metadata index tree,
     * in which each node represents a path and a location in the HDF5 virtual file hierarchy.
     */
    void xtc_tree_init(xtc_object* root_obj){
        ROOT_NODE = new_xtc_node(root_obj);
        ROOT_NODE->parent = NULL;
        (*root_obj).tree_node = (void*)ROOT_NODE;
    }
    void xtc_tree_init(int fd){
        xtc_object* root_obj = (xtc_object*)calloc(1, sizeof(xtc_object));
        root_obj->fd = fd;
        root_obj->obj_path_abs = strdup("/");
        xtc_h5token_new((xtc_token_t**)&(root_obj->obj_token), 16);
        ROOT_NODE = new_xtc_node(root_obj);
        ROOT_NODE->parent = NULL;
    }

    int xtc_tree_node_add(xtc_node* node){
        return add_xtc_node(ROOT_NODE, node);
    }

    xtc_node* xtc_tree_node_find(const char* target_path){
        return find_xtc_node(ROOT_NODE, target_path);
    }
    xtc_node* xtc_tree_node_find(string* target_path_in){
        return xtc_tree_node_find(target_path_in->c_str());
    }
    xtc_node* get_root(){
        return ROOT_NODE;
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
    NamesLookup* namesLookup(){
        return &_namesLookup; //map
    }

    void set_current_dgram(Dgram* dgram){
        assert(dgram);
        this->_current_dgram = dgram;
        printf("set_current_dgram: resulting id = %llu\n", this->_current_dgram->time.value());
    }

    /** Find the dgram_info object by dgram_id from the dgram index map.
     *  Important info fields include dgram offset in the xtc2 file, transition type and presetned dataset type.
     */
    int find_index(uint64_t dgram_id, xtc_dgram_info* dgram_info_out){
        assert(dgram_info_out);
        auto it = this->_index_map->find(dgram_id);
        if(it == _index_map->end()){
            printf("Dgram_id not exist! id = 0x%llx\n", dgram_id);
            return -1;
        } else {
            (*dgram_info_out).dgram_id = dgram_id;
            (*dgram_info_out).dgram_offset = it->second.dgram_offset;
            (*dgram_info_out).isL1 = it->second.isL1;
            (*dgram_info_out).nonL1Dgram = it->second.nonL1Dgram;
            (*dgram_info_out).ds_type = it->second.ds_type;
            return 0;
        }
    }

    /** Read the smd file and fill the dgram index in the map
     *  @param index_fd the file descriptor of the opened index file, which has the extension "smd".
     */
    int load_index(int index_fd){
        XtcFileIterator iter(index_fd, 0x4000000);
        unordered_map<uint64_t, xtc_dgram_info>* index_map = new unordered_map<uint64_t, xtc_dgram_info>;
        Dgram* smd_dgram;
        unsigned nevent=0;
        SmdIter smditer;
        printf("smditer.offset = 0x%llx\n", smditer.offset);
        while ((smd_dgram = iter.next())) {
            smditer.reset();
            smditer.iterate(&(smd_dgram->xtc));

            xtc_dgram_info info;
            info.dgram_id = smd_dgram->time.value();
            info.dgram_offset = smditer.offset;

            if(smditer.offset== (uint64_t)(int64_t)-1 ){//Non-L1 events, store in memory
                printf("smd_dgram.dgram_id = 0x%llx, smd_dgram->xtc.extent = %d, smd_dgram->xtc.sizeofPayload() = %d\n",
                        smd_dgram->time.value(), smd_dgram->xtc.extent, smd_dgram->xtc.sizeofPayload());
                size_t nonL1_dgram_size = sizeof(Dgram) + smd_dgram->xtc.extent;

                Dgram* nonL1_dgram = (Dgram*)calloc(1, nonL1_dgram_size);
                size_t memcpy_size =  sizeof(*smd_dgram) + smd_dgram->xtc.sizeofPayload();

                printf("calloc size = %d, memcpy size = %d \n", nonL1_dgram_size, memcpy_size);
                //copy header??
                //From Smd::generate
                memcpy(nonL1_dgram, smd_dgram, sizeof(*smd_dgram));
                //copy payload
                memcpy(nonL1_dgram->xtc.payload(), smd_dgram->xtc.payload(), smd_dgram->xtc.sizeofPayload());
                assert(nonL1_dgram->time.value() == smd_dgram->time.value());
                assert(0 == memcmp(nonL1_dgram->xtc.payload(), smd_dgram->xtc.payload(), smd_dgram->xtc.sizeofPayload()));

                info.isL1 = 0;
                info.ds_type = DS_NON_L1;
                info.nonL1Dgram = (void*) nonL1_dgram;
                printf("load_index: dgram_id = %llu, sizeofPayload = %d\n", info.dgram_id, smd_dgram->xtc.sizeofPayload());
            } else { //L1 events
                info.isL1 = 1;
                info.nonL1Dgram = NULL;
                info.ds_type = DS_L1;
                //info.dgram_size = smditer.dgramSize;
                printf("load_index: dgram_id = %llu, dgram_offset = 0x%llx\n", info.dgram_id, info.dgram_offset);

            }
            (*index_map)[info.dgram_id] = info;
        }

        this->_index_map = index_map;
        return 0;
    }

    /**
     * Check if the current dgram is the target, if not, load dgram from xtc2 file.
     */
    int check_update_dgram(int fd, xtc_dgram_info* dg_info, Dgram* current_dgram_in_out){
        assert(dg_info && current_dgram_in_out);
        static const unsigned bigdgBufferSize = 0x4000000;
        DEBUG_PRINT
        if((current_dgram_in_out)->time.value() != dg_info->dgram_id){
            printf("dgram_id(0x%llx) doesn't match current one(0x%llx), load dgram ...\n",
                    dg_info->dgram_id, (current_dgram_in_out)->time.value());

            if(dg_info->ds_type == DS_NON_L1 || dg_info->dgram_offset <= 0){//Non-L1 dgrams
                DEBUG_PRINT
                assert(dg_info->isL1 == 0 && dg_info->nonL1Dgram);
                *current_dgram_in_out = *(Dgram*)(dg_info->nonL1Dgram);
                //verify
                printf("Checking current_dgram_in_out.xtc.extent = %d, new dgram_id = 0x%llx, sizeofPayload = %d\n",
                        current_dgram_in_out->xtc.extent, current_dgram_in_out->time.value(), current_dgram_in_out->xtc.sizeofPayload());


            } else if (dg_info->ds_type == DS_L1){ // L1 dgrams
                //move cursor to the where the offset points to
                DEBUG_PRINT
                if (lseek(fd, (off_t)(dg_info->dgram_offset), SEEK_SET) < 0) {
                    printf("lseek error\n");
                    return -1;
                }

                //Read dgram head.
                if (::read(fd, current_dgram_in_out, sizeof(Dgram)) == 0) {//read from the offset to bigdg
                    printf("Data dgram header read error\n");
                    return -1;
                }

                //Check dgram size
                unsigned bigdgsize = sizeof(Dgram)+current_dgram_in_out->xtc.extent;
                if (bigdgsize>bigdgBufferSize) {
                    printf("Big dgram too large %s\n",bigdgsize);
                    exit(-1);
                }
                // why no to move a sizeof(Dgram) offset???
                //Read dgram data
                if (::read(fd, current_dgram_in_out->xtc.payload(), current_dgram_in_out->xtc.extent) == 0) {
                    printf("Big dgram payload read error\n");
                    exit(-1);
                }
            } else {
                DEBUG_PRINT
                printf("Wrong type: should only be DS_NON_L1(1) or DS_L1(2). type = %d\n", dg_info->ds_type);
                return -1;
            }

            DEBUG_PRINT
            printf(" done. Current dgram id = 0x%llx\n", (current_dgram_in_out)->time.value());
            return 0;
        }
        return 0;
    }

    int get_fd(){
        return _fd;
    }

    Dgram* _current_dgram;

private:
    int _fd;
    ItType iterator_type;
    NamesLookup _namesLookup;

    unordered_map<uint64_t, xtc_dgram_info>* _index_map;

    xtc_object* extern_helper;
    xtc_node* ROOT_NODE;
    int _scan_index; //mark which token to compare

    int _index_increment(){
        _scan_index++;
        return _scan_index;
    }
    int _index_decrement(){
        _scan_index--;
        return _scan_index;
    }
};

/**
 * Read and return the whole dataset for HDF5 dataset_read() call.
 */
size_t dataset_read_all(xtc_object* obj, void* buf_out){
    size_t read_size = 0;
    assert(obj->location && obj->ds_info &&
            (obj->obj_type == XTC_DS || obj->obj_type == XTC_TIME_DS));

    assert(obj->location->dbgiter);
    assert(obj->ds_info);

    DebugIter* dbgit = (DebugIter*)(obj->location->dbgiter);

    if(obj->ds_info->ds_type == DS_TIMESTAMP){//timestamps dataset

        vector<double>* time_stamps_ds = (vector<double>*) obj->data;
        read_size = time_stamps_ds->size()* sizeof(double);

        for(int i = 0; i < time_stamps_ds->size(); i++){
            double ts = time_stamps_ds->at(i);
            memcpy((char*)buf_out + (i * sizeof(double)), &ts, sizeof(double));
            printf("dataset_read_all: timestamp = %lf\n", ts);
        }
    } else if(obj->ds_info->ds_type == DS_NON_L1){//read from in-memory non-L1 dgrams.
//        printf("%s:%d: obj->ds_info->ds_type = %d, dgram_info->ds_type = %d, dgram_info->nonL1Dgram = %p\n",
//                __func__, __LINE__, obj->ds_info->ds_type, obj->ds_info->data_handle->dgram_info->ds_type, obj->ds_info->data_handle->dgram_info->nonL1Dgram);

        read_size = dbgit->get_data(obj, obj->ds_info->total_pixel_cnt, buf_out);
    } else if(obj->ds_info->ds_type == DS_L1){//regular xtc2 dataset
//        printf("%s:%d: obj->ds_info->ds_type = %d, obj->ds_info->data_handle->dgram_info->ds_type = %d\n",
//                __func__, __LINE__, obj->ds_info->ds_type, obj->ds_info->data_handle->dgram_info->ds_type);

        Xtc* xtc = (Xtc*)(obj->ds_info->data_handle->xtc_ptr);
        size_t read_size = dbgit->get_data(obj, obj->ds_info->total_pixel_cnt, buf_out);
        printf("data = [%s], data = %p, read_size = %d\n", (char*)(buf_out), buf_out, read_size);

    } else {
        printf("Unsupported ds_type: %d\n", obj->ds_info->ds_type);
        return -1;
    }

    DEBUG_PRINT
    return read_size;
}

void print_path(vector<string>vec)
{
    if(vec.size()==0)
        return;
    printf(" path = ");
    for(vector<string>::iterator it = vec.begin(); it != vec.end(); ++it){
        printf("/%s", (*it).c_str());
    }
}

vector<string> str_tok(const char* str, const char* delimiters_str){
    assert(str && delimiters_str);
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

int verifyDgram(Dgram* dgram){

    return -1;
}

/**
 * The entry point of iterating the xtc2 file, it read the file,
 * build metadata index tree and dgram index map for later faster access.
 */
xtc_object* iterate_list_all(int fd, int index_fd){
    static const unsigned bigdgBufferSize = 0x4000000;

    XtcFileIterator* iter = new XtcFileIterator(fd, bigdgBufferSize);

    DebugIter* dbgiter = new DebugIter();

    bool debugprint = true;

    dbgiter->load_index(index_fd);

    dbgiter->_current_dgram = (Dgram*)malloc(bigdgBufferSize);
    //location init
    dbgiter->CURRENT_LOCATION = (xtc_location*)calloc(1, sizeof(xtc_location));
    dbgiter->CURRENT_LOCATION->fd = fd;
    dbgiter->CURRENT_LOCATION->fileIter = (void*)(iter);
    dbgiter->CURRENT_LOCATION->dbgiter = (void*)(dbgiter);
    dbgiter->CURRENT_LOCATION->dg = NULL;

    dbgiter->index_init();//-1
    dbgiter->set_iterator_type(DebugIter::LIST_ALL);
    dbgiter->_current_dgram = iter->next();//first dg, for configure transition.
    dbgiter->CURRENT_LOCATION->dg = (void*) dbgiter->_current_dgram;

    xtc_object* head_obj = xtc_obj_new(fd, iter, dbgiter, dbgiter->_current_dgram, "/");
    head_obj->obj_type = XTC_HEAD;
    dbgiter->xtc_tree_init(head_obj);

    string candidate_str = string(TransitionId::name(dbgiter->_current_dgram->service()));
    dbgiter->append_token(candidate_str);
    dbgiter->iterate(&(dbgiter->_current_dgram->xtc));
    dbgiter->pop_token();

    int L1accept_cnt = 0;
    bool isConfigureTransition = true;

    while (dbgiter->_current_dgram = iter->next()) {//each data item in the file
        string candidate_str = string(TransitionId::name(dbgiter->_current_dgram->service()));
        dbgiter->append_token(candidate_str);
        xtc_object* xtc_obj = xtc_obj_new(fd, iter, dbgiter, dbgiter->_current_dgram, dbgiter->get_current_path().c_str());
        dbgiter->xtc_tree_node_add(new_xtc_node(xtc_obj));

        DEBUG_PRINT
        Xtc* xtc = &(dbgiter->_current_dgram->xtc);
        if (debugprint) {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            printf("iterate_list_all(): xtc = %p, shapesdata = %p\n", xtc, &shapesdata);
            dbgiter->iterate(xtc);
        }

        dbgiter->pop_token();
    }
    printf("\n\n\n\n");

    dbgiter->xtc_tree_print();
    printf("\n\n\n\n");
    return head_obj;
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
    xtc_h5token_new((xtc_token_t**)&(ret->obj_token), 16);
    return ret;
}

xtc_object* xtc_obj_new(xtc_location* location, const char* obj_path_abs){
    assert(location);
    xtc_object* obj = xtc_obj_new(location->fd,
            location->fileIter, location->dbgiter, location->dg, obj_path_abs);
    return obj;
}

unsigned long xtc_h5token_new(xtc_token_t** token, unsigned int h5_token_size) {
    if(!*token)
        *token = (xtc_token_t*)calloc(1, sizeof(xtc_token_t));

    struct timeval tp;
    gettimeofday(&tp, NULL);
    int32_t tval = (int32_t)((1000000 * tp.tv_sec) + tp.tv_usec);
    int n = sizeof(int32_t)/sizeof(uint8_t);
    for(int i = 0; i < h5_token_size; i++){//16
        if(i < n)
            (*token)->__data[i] = *(uint8_t*)(&tval + i * sizeof(uint8_t));
        else
            (*token)->__data[i] = 0;
    }
    return tval;
}

void cc_extern_test_root(void* root_obj){
    DEBUG_PRINT
    return extern_test_root((xtc_object*)root_obj);
}

EXTERNC void extern_test_root(xtc_object* root_obj){
    assert(root_obj);
    assert(root_obj->location);
    printf("%s:%d:  xtc_obj = %p, dbg = %p, dbg->get_root() = %p\n",
            __func__, __LINE__, root_obj, root_obj->location->dbgiter,
            ((DebugIter*)(root_obj->location->dbgiter))->get_root());
}

EXTERNC xtc_object* xtc_obj_find(xtc_object* root_obj, const char* path){
    assert(root_obj && path);
    assert(root_obj->location);

    if(strcmp(path, "/") == 0){
        return root_obj;
    }

    assert(((DebugIter*)(root_obj->location->dbgiter))->get_root());
    xtc_node* node = ((DebugIter*)(root_obj->location->dbgiter))->xtc_tree_node_find(path);

    if(node)
        return node->my_obj;
    else
        return NULL;
}

EXTERNC xtc_object* xtc_file_open(const char* file_path){
    int fd = open(file_path, O_RDONLY);//open xtc2 file
    string fname = string(file_path);
    vector<string> tokens = str_tok(file_path, ".");
    assert(tokens.size() == 2 && tokens[1].compare("xtc2") == 0);

    string smd_fname(tokens[0]);
    smd_fname += ".smd";
    smd_fname += ".xtc2";
    int index_fd = open(smd_fname.c_str(), O_RDONLY);//open smd file
    printf("fd = %d, index_fd = %d\n", fd, index_fd);
    xtc_object* head_obj = iterate_list_all(fd, index_fd);
    return head_obj;//contains a pointer to root node.
}

EXTERNC void xtc_file_close(xtc_object* head){
    assert(head);
    //assert(head->location);
//    if(head->location->dbgiter){
//        DebugIter* dbgiter = (DebugIter*) head->location->dbgiter;
//        if(dbgiter)
//            delete dbgiter;
//    }
//    if(head->location->dg){
//        Dgram* dg = (Dgram*) head->location->dg;
//        if(dg)
//            delete dg;
//    }

    close(head->fd);
    //delete helper->dbgiter;
    //delete (Dgram*)(helper->target_it);
//    free(helper);
}

EXTERNC xtc_object** xtc_get_children_list(xtc_object* group_in, int* num_out){
    return get_children_obj(group_in, num_out);//NULL if no children
}

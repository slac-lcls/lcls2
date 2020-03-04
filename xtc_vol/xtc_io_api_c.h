/*
 * xtc_io_api.h
 *
 *  Created on: Jan 20, 2020
 *      Author: tonglin
 */


#ifndef XTC_IO_API_H_
#define XTC_IO_API_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

#define PATH_LEN_MAX 100

typedef void* xtc_func_t;

//EXTERNC typedef enum TARGET_TYPE{
//    FILE,
//    GROUP,
//    DS,
//    OBJ
//} target_type;

/* The xtc VOL info object */
EXTERNC typedef enum XTC_Data_type{//definition is same with xtc DataType.
    UINT8, UINT16, UINT32, UINT64,
    INT8, INT16, INT32, INT64,
    FLOAT, DOUBLE,
    CHARSTR,
    ENUMVAL, ENUMDICT
}xtc_data_type;

EXTERNC typedef enum XTC_Object_type{
    XTC_FILE,
    XTC_HEAD,//root group
    XTC_GROUP,
    XTC_DS,
    XTC_LEAVE,
}xtc_object_type_t;

EXTERNC typedef enum EXECUTION_RETURN{
    SUCCESS,
    NOT_FOUND,
    RESERVED_1
}exec_ret;

//defined identical to H5O_token_t
typedef struct XTC_token {
    uint8_t __data[16];
} xtc_token_t;

EXTERNC typedef struct Xtc_Location {
    int fd;
    void* fileIter; //XtcFileIterator* const
    void* dbgiter; //DebugIter* const
    void* dg; ////Dgram* dg = fileIter.next()
    void* root_node; //used in xtc_io_api.cc.
}xtc_location;

EXTERNC typedef unsigned long long h5_size_t;

EXTERNC typedef struct XTC_DS_info {
    xtc_data_type type; //data type
    int dim_cnt;// for rank in H5Screate(rank, ), n of dimensions of dataspace.
    h5_size_t* current_dims; //1D array of size cnt_dimes,
    h5_size_t* maximum_dims; //Optional, maximum size of each dimension
    int element_size;//similar to sizeof(type)
    int element_cnt;
    void* data_ptr;
}xtc_ds_info;

EXTERNC typedef struct XTC_C_API_HELPER {
    int fd;
    int ref_cnt;
    char* obj_path_abs;
    char* obj_path_seg;
    exec_ret exec_state;//enum
    xtc_object_type_t obj_type;

//    void* target_it;//
//    void* dbgiter;
//    void* fileIterator;//XtcFileIterator
    xtc_location* location;
    xtc_ds_info* ds_info;//info of dataset. Null for non-data-set objects.
    void* obj_token;//h5token, used as an UUID for this obj.
    void* tree_node;//points to it's node location in the tree
    //void* root_obj;//type xtc_obj, all the same for every obj, all point to the root.
    //void* xtc_file_iterator;
}xtc_object;

//C API definition starts here
EXTERNC unsigned long xtc_h5token_new(xtc_token_t** token_in_out, unsigned int h5_token_size);

EXTERNC xtc_object* xtc_file_open(const char* path);//returns xtc_c_helper*
EXTERNC xtc_func_t xtc_it_open(void* param);// takes xtc_c_helper*
EXTERNC void xtc_file_close(xtc_object* helper);

//the parameter obj can be any valid obj in the tree.
EXTERNC xtc_object* xtc_obj_find(xtc_object* obj, const char* path);
EXTERNC xtc_object** xtc_get_children_list(xtc_object* group, int* num_out);
EXTERNC void extern_test_root(xtc_object* root_obj);
//used by  group_open() in XTC2 VOL
//EXTERNC xtc_object* outdated_xtc_path_search(xtc_object* file, char* path);



//C API definition ends here

#endif /* XTC_IO_API_H_ */

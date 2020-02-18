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
EXTERNC typedef enum XTC_Object_type{
    XTC_FILE,
    XTC_ROOT_GROUP,
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
}xtc_location;

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
    void* obj_token;//h5token, used as an UUID for this obj.

    //void* xtc_file_iterator;
}xtc_object;

//C API definition starts here
EXTERNC unsigned long xtc_token_new(xtc_token_t* token, unsigned int h5_token_size);

EXTERNC xtc_object* xtc_file_open(char* path);//returns xtc_c_helper*
EXTERNC xtc_func_t xtc_it_open(void* param);// takes xtc_c_helper*
EXTERNC void xtc_file_close(xtc_object* helper);

//used by  group_open() in XTC2 VOL
EXTERNC xtc_object* xtc_path_search(xtc_object* file, char* path);



//C API definition ends here

#endif /* XTC_IO_API_H_ */

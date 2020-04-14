/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* Header files needed */
/* Do NOT include private HDF5 files here! */
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Public HDF5 file */
#include <hdf5.h>
//#include <H5LTpublic.h>

/* This connector's header */
#include "xtc_vol.h"

#include "xtc_io_api_c.h"

#define DEBUG_PRINT //printf("%s():%d\n", __func__, __LINE__);
/**********/
/* Macros */
/**********/

/* Whether to display log messge when callback is invoked */
/* (Uncomment to enable) */
/* #define ENABLE_XTC_LOGGING */

/* Hack for missing va_copy() in old Visual Studio editions
 * (from H5win2_defs.h - used on VS2012 and earlier)
 */
#if defined(_WIN32) && defined(_MSC_VER) && (_MSC_VER < 1800)
#define va_copy(D,S)      ((D) = (S))
#endif

/************/
/* Typedefs */
/************/

/* The xtc VOL info object */
//typedef enum Object_type{
//    XTC_FILE,
//    XTC_GROUP,
//    XTC_DS,
//    XTC_LEAVE,
//}XTC_Object_type;

typedef struct H5VL_xtc_t {
    //hid_t  under_vol_id;//Keep as placeholder, but not used.        /* ID for underlying VOL connector */
    //void   *under_object;//Keep as placeholder, but not used.       /* Info object for underlying VOL connector */

    char *obj_path;
    xtc_object_type_t xtc_obj_type;
    xtc_object* xtc_obj;//one per file.
    /*    Points to a Xtc/Dgram object, used to keep the handle for iterate(xtc_it) in xtc_io_api.cc
     *
     */
} H5VL_xtc_t;

/* The xtc VOL wrapper context */
typedef struct H5VL_xtc_wrap_ctx_t {
    hid_t under_vol_id;         /* VOL ID for under VOL */
    void *under_wrap_ctx;       /* Object wrapping context for under VOL */
} H5VL_xtc_wrap_ctx_t;


/********************* */
/* Function prototypes */
/********************* */

/* Helper routines */

static herr_t H5VL_xtc_file_specific_reissue(void *obj, hid_t connector_id,
    H5VL_file_specific_t specific_type, hid_t dxpl_id, void **req, ...);
static herr_t H5VL_xtc_request_specific_reissue(void *obj, hid_t connector_id,
    H5VL_request_specific_t specific_type, ...);
static herr_t H5VL_xtc_link_create_reissue(H5VL_link_create_type_t create_type,
    void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id,
    hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req, ...);
static H5VL_xtc_t *H5VL_xtc_new_obj(xtc_object* helper_in);
static herr_t H5VL_xtc_free_obj(H5VL_xtc_t *obj);

/* "Management" callbacks */
static herr_t H5VL_xtc_init(hid_t vipl_id);
static herr_t H5VL_xtc_term(void);

/* VOL info callbacks */
static void *H5VL_xtc_info_copy(const void *info);
static herr_t H5VL_xtc_info_cmp(int *cmp_value, const void *info1, const void *info2);
static herr_t H5VL_xtc_info_free(void *info);
static herr_t H5VL_xtc_info_to_str(const void *info, char **str);
static herr_t H5VL_xtc_str_to_info(const char *str, void **info);

/* VOL object wrap / retrieval callbacks */
static void *H5VL_xtc_get_object(const void *obj);
static herr_t H5VL_xtc_get_wrap_ctx(const void *obj, void **wrap_ctx);
static void *H5VL_xtc_wrap_object(void *base_obj, H5I_type_t obj_type,
    void *wrap_ctx);
static void *H5VL_xtc_unwrap_object(void *base_obj);
static herr_t H5VL_xtc_free_wrap_ctx(void *base_obj);

/* Attribute callbacks */
static void *H5VL_xtc_attr_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t dxpl_id, void **req);
static void *H5VL_xtc_attr_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t aapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_attr_read(void *attr, hid_t mem_type_id, void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_attr_write(void *attr, hid_t mem_type_id, const void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_attr_get(void *obj, H5VL_attr_get_t get_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_attr_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_attr_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_attr_optional(void *obj, H5VL_attr_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_attr_close(void *attr, hid_t dxpl_id, void **req);

/* Dataset callbacks */
static void *H5VL_xtc_dataset_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t lcpl_id, hid_t type_id, hid_t space_id, hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id, void **req);
static void *H5VL_xtc_dataset_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t dapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_dataset_read(void *dset, hid_t mem_type_id, hid_t mem_space_id,
                                    hid_t file_space_id, hid_t plist_id, void *buf, void **req);
static herr_t H5VL_xtc_dataset_write(void *dset, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t plist_id, const void *buf, void **req);
static herr_t H5VL_xtc_dataset_get(void *dset, H5VL_dataset_get_t get_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_dataset_specific(void *obj, H5VL_dataset_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_dataset_optional(void *obj, H5VL_dataset_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_dataset_close(void *dset, hid_t dxpl_id, void **req);

/* Datatype callbacks */
static void *H5VL_xtc_datatype_commit(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id, hid_t tapl_id, hid_t dxpl_id, void **req);
static void *H5VL_xtc_datatype_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t tapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_datatype_get(void *dt, H5VL_datatype_get_t get_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_datatype_specific(void *obj, H5VL_datatype_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_datatype_optional(void *obj, H5VL_datatype_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_datatype_close(void *dt, hid_t dxpl_id, void **req);

/* File callbacks */
static void *H5VL_xtc_file_create(const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id, hid_t dxpl_id, void **req);
static void *H5VL_xtc_file_open(const char *name, unsigned flags, hid_t fapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_file_get(void *file, H5VL_file_get_t get_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_file_specific(void *file, H5VL_file_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_file_optional(void *obj, H5VL_file_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_file_close(void *file, hid_t dxpl_id, void **req);

/* Group callbacks */
static void *H5VL_xtc_group_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id, hid_t dxpl_id, void **req);
static void *H5VL_xtc_group_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t gapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_group_get(void *obj, H5VL_group_get_t get_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_group_specific(void *obj, H5VL_group_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_group_optional(void *obj, H5VL_group_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_group_close(void *grp, hid_t dxpl_id, void **req);

/* Link callbacks */
static herr_t H5VL_xtc_link_create(H5VL_link_create_type_t create_type, void *obj, const H5VL_loc_params_t *loc_params, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_link_copy(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_link_move(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_link_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_get_t get_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_link_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_link_optional(void *obj, H5VL_link_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments);

/* Object callbacks */
static void *H5VL_xtc_object_open(void *obj, const H5VL_loc_params_t *loc_params, H5I_type_t *opened_type, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_object_copy(void *src_obj, const H5VL_loc_params_t *src_loc_params, const char *src_name, void *dst_obj, const H5VL_loc_params_t *dst_loc_params, const char *dst_name, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_xtc_object_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_get_t get_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_object_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments);
static herr_t H5VL_xtc_object_optional(void *obj, H5VL_object_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments);

static herr_t H5VL_xtc_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl, const struct H5VL_class_t **conn_cls);
static herr_t H5VL_xtc_introspect_opt_query(void *obj, H5VL_subclass_t cls, int opt_type, hbool_t *supported);

/* Async request callbacks */
//static herr_t H5VL_xtc_request_wait(void *req, uint64_t timeout, H5ES_status_t *status);
//static herr_t H5VL_xtc_request_notify(void *obj, H5VL_request_notify_t cb, void *ctx);
//static herr_t H5VL_xtc_request_cancel(void *req);
//static herr_t H5VL_xtc_request_specific(void *req, H5VL_request_specific_t specific_type, va_list arguments);
//static herr_t H5VL_xtc_request_optional(void *req, va_list arguments);
//static herr_t H5VL_xtc_request_free(void *req);

/* Blob callbacks */
static herr_t H5VL_xtc_blob_put(void *obj, const void *buf, size_t size, void *blob_id, void *ctx);
static herr_t H5VL_xtc_blob_get(void *obj, const void *blob_id, void *buf, size_t size, void *ctx);
//static herr_t H5VL_xtc_blob_specific(void *obj, void *blob_id, H5VL_blob_specific_t specific_type, va_list arguments);

static herr_t H5VL_xtc_token_cmp(void *obj, const H5O_token_t *token1, const H5O_token_t *token2, int *cmp_value);
static herr_t H5VL_xtc_token_to_str(void *obj, H5I_type_t obj_type, const H5O_token_t *token, char **token_str);
static herr_t H5VL_xtc_token_from_str(void *obj, H5I_type_t obj_type, const char *token_str, H5O_token_t *token);

/*******************/
/* Local variables */
/*******************/

/* XTC VOL connector class struct */
const H5VL_class_t H5VL_xtc_g = {//H5VL_xtc_g
        XTC_VOL_VERSION,                          /* version      */
    (H5VL_class_value_t)XTC_VOL_VALUE,        /* value        */
    XTC_VOL_NAME,                             /* name         */
    0,                                              /* capability flags */
    H5VL_xtc_init,                         /* initialize   */
    H5VL_xtc_term,                         /* terminate    */
    {                                           /* info_cls */
        0, //sizeof(H5VL_xtc_info_t),           /* size    */
        NULL, //H5VL_xtc_info_copy,                /* copy    */
        NULL, //H5VL_xtc_info_cmp,                 /* compare */
        NULL, //H5VL_xtc_info_free,                /* free    */
        NULL,//H5VL_xtc_info_to_str,              /* to_str  */
        NULL//H5VL_xtc_str_to_info               /* from_str */
    },
    {                                           /* wrap_cls */
        NULL, //H5VL_xtc_get_object,               /* get_object   */
        NULL,//H5VL_xtc_get_wrap_ctx,             /* get_wrap_ctx */
        NULL,//H5VL_xtc_unwrap_object,            /* unwrap_object */
        NULL,//H5VL_xtc_wrap_object,              /* wrap_object  */
        NULL,//H5VL_xtc_free_wrap_ctx             /* free_wrap_ctx */
    },
    {                                           /* attribute_cls */
        H5VL_xtc_attr_create,              /* create */
        H5VL_xtc_attr_open,                /* open */
        H5VL_xtc_attr_read,                /* read */
        H5VL_xtc_attr_write,               /* write */
        H5VL_xtc_attr_get,                 /* get */
        H5VL_xtc_attr_specific,            /* specific */
        H5VL_xtc_attr_optional,            /* optional */
        H5VL_xtc_attr_close                /* close */
    },
    {                                           /* dataset_cls */
        H5VL_xtc_dataset_create,           /* create */
        H5VL_xtc_dataset_open,             /* open */
        H5VL_xtc_dataset_read,             /* read */
        H5VL_xtc_dataset_write,            /* write */
        H5VL_xtc_dataset_get,              /* get */
        H5VL_xtc_dataset_specific,         /* specific */
        H5VL_xtc_dataset_optional,         /* optional */
        H5VL_xtc_dataset_close             /* close */
    },
    {                                           /* datatype_cls */
        H5VL_xtc_datatype_commit,          /* commit */
        H5VL_xtc_datatype_open,            /* open */
        H5VL_xtc_datatype_get,             /* get_size */
        H5VL_xtc_datatype_specific,        /* specific */
        H5VL_xtc_datatype_optional,        /* optional */
        H5VL_xtc_datatype_close            /* close */
    },
    {                                           /* file_cls */
        H5VL_xtc_file_create,              /* create */
        H5VL_xtc_file_open,                /* open */
        H5VL_xtc_file_get,                 /* get */
        H5VL_xtc_file_specific,            /* specific */
        H5VL_xtc_file_optional,            /* optional */
        H5VL_xtc_file_close                /* close */
    },
    {                                           /* group_cls */
        H5VL_xtc_group_create,             /* create */
        H5VL_xtc_group_open,               /* open */
        H5VL_xtc_group_get,                /* get */
        H5VL_xtc_group_specific,           /* specific */
        H5VL_xtc_group_optional,           /* optional */
        H5VL_xtc_group_close               /* close */
    },
    {                                           /* link_cls */
        H5VL_xtc_link_create,              /* create */
        H5VL_xtc_link_copy,                /* copy */
        H5VL_xtc_link_move,                /* move */
        H5VL_xtc_link_get,                 /* get */
        H5VL_xtc_link_specific,            /* specific */
        H5VL_xtc_link_optional             /* optional */
    },
    {                                           /* object_cls */
        H5VL_xtc_object_open,              /* open */
        H5VL_xtc_object_copy,              /* copy */
        H5VL_xtc_object_get,               /* get */
        H5VL_xtc_object_specific,          /* specific */
        H5VL_xtc_object_optional           /* optional */
    },
    {
        H5VL_xtc_introspect_get_conn_cls, //get_conn_cls
        H5VL_xtc_introspect_opt_query, //opt_query
    },
    {                                           /* request_cls */
        NULL, //        H5VL_xtc_request_wait,             /* wait */
        NULL,         //        H5VL_xtc_request_notify,           /* notify */
        NULL,         //        H5VL_xtc_request_cancel,           /* cancel */
        NULL,         //        H5VL_xtc_request_specific,         /* specific */
        NULL,         //        H5VL_xtc_request_optional,         /* optional */
        NULL,         //        H5VL_xtc_request_free              /* free */
    },

    {                                           /* blob_cls */
        H5VL_xtc_blob_put,                 /* put */
        H5VL_xtc_blob_get,                 /* get */
        //H5VL_xtc_blob_specific,            /* specific */
        NULL,                                        /* optional */
    },
    {//token
            H5VL_xtc_token_cmp,
            H5VL_xtc_token_to_str,
            H5VL_xtc_token_from_str,
    },
    NULL                                        /* optional */
};

/* The connector identification number, initialized at runtime */
static hid_t H5VL_XTC_g = H5I_INVALID_HID;

H5PL_type_t H5PLget_plugin_type(void) {return H5PL_TYPE_VOL;}
const void *H5PLget_plugin_info(void) {return &H5VL_xtc_g;}

hid_t type_convert(xtc_data_type xtc_type);

unsigned long new_xtc_h5token(H5O_token_t* token) {
    assert(token);
    return xtc_h5token_new((xtc_token_t**)(&token), H5O_MAX_TOKEN_SIZE);
}

int xtc_h5token_cmp(H5O_token_t* t1, H5O_token_t* t2){
    assert(t1&&t2);
    for(int i = 0; i < H5O_MAX_TOKEN_SIZE; i++){
        if(t1->__data[i] != t2->__data[i])
            return -1;
    }
    return 0;
}

char* xtc_h5token_to_str(const H5O_token_t* t){
    assert(t);
    // token sample: 123-46-789-1
    char* str = (char*)calloc(1, 18 * sizeof(char));
    sprintf(str, "%d-%d-%d-%d", t->__data[0], t->__data[1], t->__data[2], t->__data[3]);
    //printf("test token_to_str: %s\n", str);
    return str;
}

int token_copy(H5O_token_t** dst_in_out, H5O_token_t* src_in){
    assert(*dst_in_out && src_in);
    for(int i = 0; i < H5O_MAX_TOKEN_SIZE; i++)
        (*dst_in_out)->__data[i] = src_in->__data[i];
    return 0;
}

H5VL_xtc_t* _internal_obj_lookup_by_name(H5VL_xtc_t *base_obj, const char* path){
    H5VL_xtc_t* ret_obj = NULL;
    assert(base_obj && path && strlen(path) > 0);
    if(strlen(path) == 1){// . or /
        //case '.' for relative path points to self.
        if(path[0] == '.'){
            assert(base_obj->xtc_obj && base_obj->xtc_obj->obj_path_abs);
            ret_obj = base_obj;
        } else if(path[0] == '/') {
            assert(strcmp(base_obj->xtc_obj->obj_path_abs, "/") == 0);
            ret_obj = base_obj;
        } else {
            assert(0 && "_internal_obj_lookup_by_name(): Not implemented.");
        }
        return ret_obj;
    }

    //assert(base_obj->xtc_obj_type = XTC_GROUP);
    char* abs_path = NULL;
    if(path[0] == '.' && path[1] == '/'){//relative path.
        int size = strlen(base_obj->xtc_obj->obj_path_abs) + strlen(path) + 3;
        abs_path = (char*)calloc(1, size*sizeof(char));
        sprintf(abs_path, "%s/%s", base_obj->obj_path, path);
    }

    if(path[0] == '/'){//absolute path.
        abs_path = (char*)path;
    }
    //printf("%s:  absolute path = %s\n", __func__, abs_path);
    DEBUG_PRINT
    xtc_object* obj = xtc_obj_find(base_obj->xtc_obj, abs_path);
    ret_obj = H5VL_xtc_new_obj(obj);
    return ret_obj;
}

H5VL_xtc_t* _object_lookup(H5VL_xtc_t *base_obj, const H5VL_loc_params_t *loc_params){
    H5VL_xtc_t* target = NULL;
    switch(loc_params->type){
        case H5VL_OBJECT_BY_SELF:
            //printf("%s:%d: loc_type = H5VL_OBJECT_BY_SELF\n", __func__, __LINE__);// "/"
            target = base_obj;
            break;

        case H5VL_OBJECT_BY_NAME:
//            printf("%s:%d: loc_type = H5VL_OBJECT_BY_NAME, name = %s, loc_obj_type = %d, base_obj_type = %d\n",
//                    __func__, __LINE__, loc_params->loc_data.loc_by_name.name, loc_params->obj_type, base_obj->xtc_obj->obj_type);
            //need to handle relative/absolute and find the correct obj
            //base_obj is the obj that's used to start on, search key is in loc_params.
            target = _internal_obj_lookup_by_name(base_obj, loc_params->loc_data.loc_by_name.name);
            break;

        case H5VL_OBJECT_BY_IDX:
            printf("%s:%d: loc_type = H5VL_OBJECT_BY_IDX\n", __func__, __LINE__);
            assert(0 && "Not implemented");
            break;

        case H5VL_OBJECT_BY_TOKEN:
            printf("%s:%d: loc_type = H5VL_OBJECT_BY_TOKEN\n", __func__, __LINE__);
            assert(0 && "Not implemented");
            break;

        default:
            printf("%s:%d: type = Unknown loc_type: %d \n", __func__, loc_params->type, __LINE__);
            break;
    }
    return target;
}



/*-------------------------------------------------------------------------
 * Function:    H5VL__xtc_new_obj
 *
 * Purpose:     Create a new xtc object for an underlying object
 *
 * Return:      Success:    Pointer to the new xtc object
 *              Failure:    NULL
 *
 * Programmer:  Quincey Koziol
 *              Monday, December 3, 2018
 *
 *-------------------------------------------------------------------------
 */
static H5VL_xtc_t *
H5VL_xtc_new_obj(xtc_object* obj_in)
{
    DEBUG_PRINT

    //assert(obj_in);
    if(!obj_in)
        return NULL;
    H5VL_xtc_t *new_obj;
    //DEBUG_PRINT
    new_obj = (H5VL_xtc_t *)calloc(1, sizeof(H5VL_xtc_t));

    //new_obj->under_vol_id = under_vol_id;

    new_obj->xtc_obj = obj_in;//use global var for now.
    new_obj->xtc_obj->ref_cnt++;
    new_obj->xtc_obj_type = obj_in->obj_type;
    assert(obj_in->obj_path_abs);
    new_obj->obj_path = obj_in->obj_path_abs;
    //new_obj->under_object = under_obj;//terminal, no under obj, points to self

    //H5Iinc_ref(vol_id);//new_obj->under_vol_id

    return new_obj;
} /* end H5VL__xtc_new_obj() */


/*-------------------------------------------------------------------------
 * Function:    H5VL__xtc_free_obj
 *
 * Purpose:     Release a xtc object
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 * Programmer:  Quincey Koziol
 *              Monday, December 3, 2018
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_free_obj(H5VL_xtc_t *obj)
{

    //assert(0 && "breakpoint");
    //hid_t err_id;

    //err_id = H5Eget_current_stack();
    assert(obj && obj->xtc_obj);
    obj->xtc_obj->ref_cnt--;
    //H5Idec_ref(obj->under_vol_id);

    //H5Eset_current_stack(err_id);
    if(obj->xtc_obj->ref_cnt == 0){
        //Do nothing, helper is freed on file_close.
    }
    DEBUG_PRINT
    free(obj);
    return 0;
} /* end H5VL__xtc_free_obj() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_register
 *
 * Purpose:     Register the xtc VOL connector and retrieve an ID
 *              for it.
 *
 * Return:      Success:    The ID for the xtc VOL connector
 *              Failure:    -1
 *
 * Programmer:  Quincey Koziol
 *              Wednesday, November 28, 2018
 *
 *-------------------------------------------------------------------------
 */
hid_t
H5VL_xtc_register(void)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    /* Singleton register the xtc VOL connector ID */
    if(H5VL_XTC_g < 0)
        H5VL_XTC_g = H5VLregister_connector(&H5VL_xtc_g, H5P_DEFAULT);

    return H5VL_XTC_g;
} /* end H5VL_xtc_register() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_init
 *
 * Purpose:     Initialize this VOL connector, performing any necessary
 *              operations for the connector that will apply to all containers
 *              accessed with the connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_init(hid_t vipl_id)
{
    //assert(0 && "breakpoint");
    DEBUG_PRINT
#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL INIT\n");
#endif

    /* Shut compiler up about unused parameter */
    vipl_id = vipl_id;

    return 0;
} /* end H5VL_xtc_init() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_term
 *
 * Purpose:     Terminate this VOL connector, performing any necessary
 *              operations for the connector that release connector-wide
 *              resources (usually created / initialized with the 'init'
 *              callback).
 *
 * Return:      Success:    0
 *              Failure:    (Can't fail)
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_term(void)
{    //assert(0 && "breakpoint");
    DEBUG_PRINT
#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL TERM\n");
#endif

    /* Reset VOL ID */
    H5VL_XTC_g = H5I_INVALID_HID;

    return 0;
} /* end H5VL_xtc_term() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_info_copy
 *
 * Purpose:     Duplicate the connector's info object.
 *
 * Returns:     Success:    New connector info object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_xtc_info_copy(const void *_info)
{
    //assert(0 && "breakpoint");
    DEBUG_PRINT
    const H5VL_xtc_info_t *info = (const H5VL_xtc_info_t *)_info;
    H5VL_xtc_info_t *new_info;
    DEBUG_PRINT
#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL INFO Copy\n");
#endif

    /* Allocate new VOL info struct for the xtc connector */
    new_info = (H5VL_xtc_info_t *)calloc(1, sizeof(H5VL_xtc_info_t));

    /* Increment reference count on underlying VOL ID, and copy the VOL info */
    //new_info->under_vol_id = info->under_vol_id;
//    H5Iinc_ref(new_info->under_vol_id);
//    if(info->under_vol_info)
//        H5VLcopy_connector_info(new_info->under_vol_id, &(new_info->under_vol_info), info->under_vol_info);
    DEBUG_PRINT
    return new_info;
} /* end H5VL_xtc_info_copy() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_info_cmp
 *
 * Purpose:     Compare two of the connector's info objects, setting *cmp_value,
 *              following the same rules as strcmp().
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_info_cmp(int *cmp_value, const void *_info1, const void *_info2)
{

    DEBUG_PRINT
    assert(0 && "breakpoint");
    const H5VL_xtc_info_t *info1 = (const H5VL_xtc_info_t *)_info1;
    const H5VL_xtc_info_t *info2 = (const H5VL_xtc_info_t *)_info2;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL INFO Compare\n");
#endif

    /* Sanity checks */
    assert(info1);
    assert(info2);

    /* Initialize comparison value */
    *cmp_value = 0;
    
    /* Compare under VOL connector classes */
    H5VLcmp_connector_cls(cmp_value, info1->under_vol_id, info2->under_vol_id);
    if(*cmp_value != 0)
        return 0;

    /* Compare under VOL connector info objects */
    H5VLcmp_connector_info(cmp_value, info1->under_vol_id, info1->under_vol_info, info2->under_vol_info);
    if(*cmp_value != 0)
        return 0;

    return 0;
} /* end H5VL_xtc_info_cmp() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_info_free
 *
 * Purpose:     Release an info object for the connector.
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_info_free(void *_info)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_info_t *info = (H5VL_xtc_info_t *)_info;
    hid_t err_id;
    DEBUG_PRINT
#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL INFO Free\n");
#endif
    DEBUG_PRINT
    err_id = H5Eget_current_stack();
    DEBUG_PRINT
    /* Release underlying VOL ID and info */
    if(info->under_vol_info)
        H5VLfree_connector_info(info->under_vol_id, info->under_vol_info);
    H5Idec_ref(info->under_vol_id);
    DEBUG_PRINT
    H5Eset_current_stack(err_id);
    DEBUG_PRINT
    /* Free xtc info object itself */
    free(info);

    return 0;
} /* end H5VL_xtc_info_free() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_info_to_str
 *
 * Purpose:     Serialize an info object for this connector into a string
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_info_to_str(const void *_info, char **str)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    const H5VL_xtc_info_t *info = (const H5VL_xtc_info_t *)_info;
    H5VL_class_value_t under_value = (H5VL_class_value_t)-1;
    char *under_vol_string = NULL;
    size_t under_vol_str_len = 0;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL INFO To String\n");
#endif

    /* Get value and string for underlying VOL connector */
    H5VLget_value(info->under_vol_id, &under_value);
    H5VLconnector_info_to_str(info->under_vol_info, info->under_vol_id, &under_vol_string);

    /* Determine length of underlying VOL info string */
    if(under_vol_string)
        under_vol_str_len = strlen(under_vol_string);

    /* Allocate space for our info */
    *str = (char *)H5allocate_memory(32 + under_vol_str_len, (hbool_t)0);
    assert(*str);

    /* Encode our info
     * Normally we'd use snprintf() here for a little extra safety, but that
     * call had problems on Windows until recently. So, to be as platform-independent
     * as we can, we're using sprintf() instead.
     */
    sprintf(*str, "under_vol=%u;under_info={%s}", (unsigned)under_value, (under_vol_string ? under_vol_string : ""));

    return 0;
} /* end H5VL_xtc_info_to_str() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_str_to_info
 *
 * Purpose:     Deserialize a string into an info object for this connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_str_to_info(const char *str, void **_info)
{
    DEBUG_PRINT
    //assert(0 && "breakpoint");
    H5VL_xtc_info_t *info;
    unsigned under_vol_value;
    const char *under_vol_info_start, *under_vol_info_end;
    hid_t under_vol_id;
    void *under_vol_info = NULL;
    DEBUG_PRINT
#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL INFO String To Info\n");
#endif
    DEBUG_PRINT
    /* Retrieve the underlying VOL connector value and info */
    sscanf(str, "under_vol=%u;", &under_vol_value);
    DEBUG_PRINT
    under_vol_id = H5VLregister_connector_by_value((H5VL_class_value_t)under_vol_value, H5P_DEFAULT);
    DEBUG_PRINT
    under_vol_info_start = strchr(str, '{');
    under_vol_info_end = strrchr(str, '}');
    DEBUG_PRINT
    assert(under_vol_info_end > under_vol_info_start);
    DEBUG_PRINT
    if(under_vol_info_end != (under_vol_info_start + 1)) {
        char *under_vol_info_str;

        under_vol_info_str = (char *)malloc((size_t)(under_vol_info_end - under_vol_info_start));
        memcpy(under_vol_info_str, under_vol_info_start + 1, (size_t)((under_vol_info_end - under_vol_info_start) - 1));
        *(under_vol_info_str + (under_vol_info_end - under_vol_info_start)) = '\0';

        H5VLconnector_str_to_info(under_vol_info_str, under_vol_id, &under_vol_info);

        free(under_vol_info_str);
    } /* end else */
    DEBUG_PRINT
    /* Allocate new xtc VOL connector info and set its fields */
    info = (H5VL_xtc_info_t *)calloc(1, sizeof(H5VL_xtc_info_t));
    info->under_vol_id = under_vol_id;
    info->under_vol_info = under_vol_info;
    DEBUG_PRINT
    /* Set return value */
    *_info = info;

    return 0;
} /* end H5VL_xtc_str_to_info() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_get_object
 *
 * Purpose:     Retrieve the 'data' for a VOL object.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_xtc_get_object(const void *obj)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    const H5VL_xtc_t *o = (const H5VL_xtc_t *)obj;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL Get object\n");
#endif

    //return H5VLget_object(o->under_object, o->under_vol_id);
    return NULL; //H5VLget_object(o, o->under_vol_id);
} /* end H5VL_xtc_get_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_get_wrap_ctx
 *
 * Purpose:     Retrieve a "wrapper context" for an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_get_wrap_ctx(const void *obj, void **wrap_ctx)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    const H5VL_xtc_t *o = (const H5VL_xtc_t *)obj;
    H5VL_xtc_wrap_ctx_t *new_wrap_ctx;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL WRAP CTX Get\n");
#endif
    DEBUG_PRINT
    /* Allocate new VOL object wrapping context for the xtc connector */
    new_wrap_ctx = (H5VL_xtc_wrap_ctx_t *)calloc(1, sizeof(H5VL_xtc_wrap_ctx_t));
    DEBUG_PRINT
    /* Increment reference count on underlying VOL ID, and copy the VOL info */
    //new_wrap_ctx->under_vol_id = o->under_vol_id;
    //H5Iinc_ref(new_wrap_ctx->under_vol_id);
    //H5VLget_wrap_ctx(o->under_object, o->under_vol_id, &new_wrap_ctx->under_wrap_ctx);

    /* Set wrap context to return */
    *wrap_ctx = NULL;//new_wrap_ctx;

    return 0;
} /* end H5VL_xtc_get_wrap_ctx() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_wrap_object
 *
 * Purpose:     Use a "wrapper context" to wrap a data object
 *
 * Return:      Success:    Pointer to wrapped object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
//static void *
//H5VL_xtc_wrap_object(void *obj, H5I_type_t obj_type, void *_wrap_ctx)
//{DEBUG_PRINT
//    assert(0 && "breakpoint");
//    H5VL_xtc_wrap_ctx_t *wrap_ctx = (H5VL_xtc_wrap_ctx_t *)_wrap_ctx;
//    H5VL_xtc_t *new_obj;
//    void *under;
//    DEBUG_PRINT
//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL WRAP Object\n");
//#endif
//    DEBUG_PRINT
//    /* Wrap the object with the underlying VOL */
//    under = H5VLwrap_object(obj, obj_type, wrap_ctx->under_vol_id, wrap_ctx->under_wrap_ctx);
//    DEBUG_PRINT
//    if(under)
//        new_obj = H5VL_xtc_new_obj(under, wrap_ctx->under_vol_id);
//    else
//        new_obj = NULL;
//    DEBUG_PRINT
//    return new_obj;
//} /* end H5VL_xtc_wrap_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_unwrap_object
 *
 * Purpose:     Unwrap a wrapped object, discarding the wrapper, but returning
 *		underlying object.
 *
 * Return:      Success:    Pointer to unwrapped object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
//static void *
//H5VL_xtc_unwrap_object(void *obj)
//{DEBUG_PRINT
//    assert(0 && "breakpoint");
////    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
////    void *under;
//    DEBUG_PRINT
//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL UNWRAP Object\n");
//#endif
//DEBUG_PRINT
//    /* Unrap the object with the underlying VOL */
////    under = H5VLunwrap_object(o->under_object, o->under_vol_id);
////
////    if(under)
////        H5VL_xtc_free_obj(o);
//
//    return obj;
//} /* end H5VL_xtc_unwrap_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_xtc_free_wrap_ctx
 *
 * Purpose:     Release a "wrapper context" for an object
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
//static herr_t
//H5VL_xtc_free_wrap_ctx(void *_wrap_ctx)
//{    assert(0 && "breakpoint");
//    H5VL_xtc_wrap_ctx_t *wrap_ctx = (H5VL_xtc_wrap_ctx_t *)_wrap_ctx;
//    hid_t err_id;
//
//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL WRAP CTX Free\n");
//#endif
//    DEBUG_PRINT
//    err_id = H5Eget_current_stack();
////
////    /* Release underlying VOL ID and wrap context */
////    if(wrap_ctx->under_wrap_ctx)
////        H5VLfree_wrap_ctx(wrap_ctx->under_wrap_ctx, wrap_ctx->under_vol_id);
////    H5Idec_ref(wrap_ctx->under_vol_id);
//    DEBUG_PRINT
//    H5Eset_current_stack(err_id);
//    DEBUG_PRINT
//    /* Free xtc wrap context object itself */
//    free(wrap_ctx);
//
//    return 0;
//} /* end H5VL_xtc_free_wrap_ctx() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_attr_create
 *
 * Purpose:     Creates an attribute on an object.
 *
 * Return:      Success:    Pointer to attribute object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_attr_create(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t type_id, hid_t space_id, hid_t acpl_id,
    hid_t aapl_id, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *attr;
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    void *under;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL ATTRIBUTE Create\n");
#endif

    return (void*)attr;
} /* end H5VL_xtc_attr_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_attr_open
 *
 * Purpose:     Opens an attribute on an object.
 *
 * Return:      Success:    Pointer to attribute object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_attr_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t aapl_id, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *attr;
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    void *under;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL ATTRIBUTE Open\n");
#endif
    return (void *)attr;
} /* end H5VL_xtc_attr_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_attr_read
 *
 * Purpose:     Reads data from attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_attr_read(void *attr, hid_t mem_type_id, void *buf,
    hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)attr;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL ATTRIBUTE Read\n");
#endif
    return ret_value;
} /* end H5VL_xtc_attr_read() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_attr_write
 *
 * Purpose:     Writes data to attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_attr_write(void *attr, hid_t mem_type_id, const void *buf,
    hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)attr;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL ATTRIBUTE Write\n");
#endif

    return ret_value;
} /* end H5VL_xtc_attr_write() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_attr_get
 *
 * Purpose:     Gets information about an attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_attr_get(void *obj, H5VL_attr_get_t get_type, hid_t dxpl_id,
    void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL ATTRIBUTE Get\n");
#endif
    return ret_value;
} /* end H5VL_xtc_attr_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_attr_specific
 *
 * Purpose:     Specific operation on attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_attr_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_attr_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    //assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;//???is obj xtc_obj or a h5vl type?
    herr_t ret_value;
    H5VL_xtc_t* target = _object_lookup(o, loc_params);

    switch(specific_type){
        case H5VL_ATTR_ITER:                         /* H5Aiterate(_by_name)                */
            //printf("specific_type = H5VL_ATTR_ITER\n");
            //assert(0 && "breakpoint");
            //assume no attributes added for xtc.
            break;
        case H5VL_ATTR_DELETE:                       /* H5Adelete(_by_name/idx)             */
        case H5VL_ATTR_EXISTS:                       /* H5Aexists(_by_name)                 */
            //printf("specific_type = H5VL_ATTR_EXISTS\n");
            break;
        case H5VL_ATTR_RENAME:
        default:
            assert(0 && "H5VL_xtc_attr_specific: Not implemented");
            break;
    }
#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL ATTRIBUTE Specific\n");
#endif

//    ret_value = H5VLattr_specific(o->under_object, loc_params, o->under_vol_id, specific_type, dxpl_id, req, arguments);

    return ret_value;
} /* end H5VL_xtc_attr_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_attr_optional
 *
 * Purpose:     Perform a connector-specific operation on an attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_attr_optional(void *obj, H5VL_attr_optional_t opt_type, hid_t dxpl_id,
        void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL ATTRIBUTE Optional\n");
#endif
    return ret_value;
} /* end H5VL_xtc_attr_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_attr_close
 *
 * Purpose:     Closes an attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1, attr not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_attr_close(void *attr, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)attr;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL ATTRIBUTE Close\n");
#endif
    return ret_value;
} /* end H5VL_xtc_attr_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_dataset_create
 *
 * Purpose:     Creates a dataset in a container
 *
 * Return:      Success:    Pointer to a dataset object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_dataset_create(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t lcpl_id, hid_t type_id, hid_t space_id,
    hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *dset;
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    void *under;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL DATASET Create\n");
#endif
    return (void *)dset;
} /* end H5VL_xtc_dataset_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_dataset_open
 *
 * Purpose:     Opens a dataset in a container
 *
 * Return:      Success:    Pointer to a dataset object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_dataset_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t dapl_id, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *dset;//iterators indicating starting point of scanning.
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    void *under;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL DATASET Open\n");
#endif
    return (void *)dset;
} /* end H5VL_xtc_dataset_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_dataset_read
 *
 * Purpose:     Reads data elements from a dataset into a buffer.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
//H5VL_dataset_read(vol_obj, mem_type_id, mem_space_id, file_space_id, dxpl_id, buf, H5_REQUEST_NULL)
static herr_t 
H5VL_xtc_dataset_read(void *dset, hid_t mem_type_id, hid_t mem_space_id,
    hid_t file_space_id, hid_t plist_id, void *buf, void **req)
{
    /*
     * mem_space_id: h5 native type
     * file_sapce_id: user set it to specify select range
     *
     * */
    herr_t ret_value;
    H5VL_xtc_t *obj = (H5VL_xtc_t *)dset;
    assert(obj->xtc_obj);
    hid_t h5_ds_type = type_convert(obj->xtc_obj->ds_info->type);
    //printf("\n h5_ds_type = %d, mem_type_id = %d, H5T_C_S1 = %d\n", h5_ds_type, mem_type_id, H5T_C_S1);

    DEBUG_PRINT
    hsize_t varsize, stride_ind0;
    H5S_sel_type mem_select_type, file_select_type;
    hsize_t npoints, npoints_mem, nblocks_file, nblocks_mem;

    //Verify mem_type_id
    hsize_t dataTypeSize = H5Tget_size(mem_type_id);

    hid_t t_class = H5Tget_class(mem_type_id);
    H5T_sign_t sign = H5Tget_sign(mem_type_id);
    if (mem_space_id != 0) {
        mem_select_type = H5Sget_select_type(mem_space_id);
    }
    //Verify mem_space_id

    if (file_space_id != 0) {
        file_select_type = H5Sget_select_type(file_space_id);
    }

    if ((mem_select_type == H5S_SEL_NONE) || (file_select_type == H5S_SEL_NONE)) {
        /* Nothing was selected, do nothing */
        return 0;
    }

    if(file_select_type == H5S_SEL_ALL){
        //assert(0 && "H5S_SEL_ALL");

        dataset_read_all(obj->xtc_obj, buf);

    } else if(file_select_type == H5S_SEL_HYPERSLABS) {

        /* Generate arrays of flattened positions for each point in the selection.
         * The H5_posMap type will have an index (posCompact), and a position in
         * file or memory space (posInSource).  The position is the element index
         * for a flattened representation of the selection.
         */

        npoints = H5Sget_select_npoints(file_space_id);
        int dim_cnt = obj->xtc_obj->ds_info->dim_cnt;
        assert(dim_cnt >= 0 && dim_cnt <= 5);
        int total_pixel_cnt = 1;
        for(int i = 0; i < dim_cnt; i++){
            total_pixel_cnt *= obj->xtc_obj->ds_info->current_dims[i];
        }

        if(total_pixel_cnt == npoints){//case H5S_SEL_ALL
            //read all data
            //assert(0 && "H5S_SEL_ALL");
            dataset_read_all(obj->xtc_obj, buf);
            //buf
        } else { //real hyperslab read
            assert(0 && "H5S_SEL_HYPERSLABS");
            hsize_t sel_block_cnt = H5Sget_select_hyper_nblocks(file_space_id);
            hsize_t *blockinfo = (hsize_t *)calloc(1, sizeof(hsize_t) * 2 * dim_cnt * sel_block_cnt);//???
            herr_t status = H5Sget_select_hyper_blocklist(file_space_id, (hsize_t)0, sel_block_cnt, blockinfo);
            assert(status >= 0);//otherwise failed.
            // <"start" coordinate>, immediately followed by <"opposite" corner coordinate>, followed by
            // the next "start" and "opposite" coordinates, until end.


        }
    } else if(file_select_type == H5S_SEL_POINTS){
        assert(0 && "H5S_SEL_POINTS");
    } else {
        assert(0 && "Unknown file_select_type");
    }



//    switch(t_class){
//        printf("mem_type_id class: H5T_INTEGER\n");
//        case H5T_INTEGER:
//            break;
//        case H5T_FLOAT:
//            printf("mem_type_id class: H5T_FLOAT\n");
//            break;
//        case H5T_STRING:
//            printf("mem_type_id class: H5T_STRING\n");
//            break;
//        case H5T_BITFIELD:
//            printf("mem_type_id class: H5T_BITFIELD\n");
//            break;
//        case H5T_OPAQUE:
//            printf("mem_type_id class: H5T_OPAQUE\n");
//            break;
//        case H5T_VLEN:
//            printf("mem_type_id class: H5T_VLEN\n");
//            break;
//        case H5T_COMPOUND:
//            printf("mem_type_id class: H5T_COMPOUND\n");
//            break;
//        case H5T_REFERENCE:
//            printf("mem_type_id class: H5T_REFERENCE\n");
//            break;
//        case H5T_ENUM:
//            printf("mem_type_id class: H5T_ENUM\n");
//            break;
//        case H5T_ARRAY:
//            printf("mem_type_id class: H5T_ARRAY\n");
//            break;
//        default:
//            break;
//    }






#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL DATASET Read\n");
#endif
    //assert(0 && "breakpoint");

    return ret_value;
} /* end H5VL_xtc_dataset_read() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_dataset_write
 *
 * Purpose:     Writes data elements from a buffer into a dataset.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_dataset_write(void *dset, hid_t mem_type_id, hid_t mem_space_id,
    hid_t file_space_id, hid_t plist_id, const void *buf, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)dset;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL DATASET Write\n");
#endif

//    ret_value = H5VLdataset_write(o->under_object, o->under_vol_id, mem_type_id, mem_space_id, file_space_id, plist_id, buf, req);
//
//    /* Check for async request */
//    if(req && *req)
//        *req = H5VL_xtc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_xtc_dataset_write() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_dataset_get
 *
 * Purpose:     Gets information about a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */

//xtc2 to HDF5 native type mapping
hid_t type_convert(xtc_data_type xtc_type){
    char* xtc_type_name = "";
    char* h5_type_name = "";
    hid_t ret;

    switch(xtc_type){
        case UINT8:
            xtc_type_name = "UINT8";
            ret =  H5T_NATIVE_UINT8;
            h5_type_name = "H5T_NATIVE_UINT8";
            break;
        case UINT16:
            xtc_type_name = "UINT16";
            ret = H5T_NATIVE_UINT16;
            h5_type_name = "H5T_NATIVE_UINT16";
            break;
        case UINT32:
            xtc_type_name = "UINT32";
            ret = H5T_NATIVE_UINT32;
            h5_type_name = "H5T_NATIVE_UINT32";
            break;
        case UINT64:
            xtc_type_name = "UINT64";
            ret = H5T_NATIVE_UINT64;
            h5_type_name = "H5T_NATIVE_UINT64";
            break;
        case INT8:
            xtc_type_name = "INT8";
            ret = H5T_NATIVE_INT8;
            h5_type_name = "H5T_NATIVE_INT8";
            break;
        case INT16:
            xtc_type_name = "INT16";
            ret = H5T_NATIVE_INT16;
            h5_type_name = "H5T_NATIVE_INT16";
            break;
        case INT32:
            xtc_type_name = "INT32";
            ret = H5T_NATIVE_INT32;
            h5_type_name = "H5T_NATIVE_INT32";
            break;
        case INT64:
            xtc_type_name = "INT64";
            ret = H5T_NATIVE_INT64;
            h5_type_name = "H5T_NATIVE_INT64";
            break;
        case FLOAT:
            xtc_type_name = "FLOAT";
            ret = H5T_NATIVE_FLOAT;
            h5_type_name = "H5T_NATIVE_FLOAT";
            break;
        case DOUBLE:
            xtc_type_name = "DOUBLE";
            ret = H5T_NATIVE_DOUBLE;
            h5_type_name = "H5T_NATIVE_DOUBLE";
            break;

        case CHARSTR:
            xtc_type_name = "CHARSTR";
            //ret = H5T_C_S1; //C-specific string datatype
            h5_type_name = "H5T_C_S1";
            ret = H5Tcopy(H5T_C_S1);
			H5Tset_size(ret, H5T_VARIABLE);
            break;

        case ENUMVAL:
            xtc_type_name = "ENUMVAL";
            ret = H5T_NATIVE_INT32;
            h5_type_name = "H5T_NATIVE_INT32";
            break;

        case ENUMDICT:
            xtc_type_name = "ENUMDICT";
            ret = H5T_NATIVE_INT32;
            h5_type_name = "H5T_NATIVE_INT32";
            break;

         default:
             printf("Unsupported type: %d\n", xtc_type);
             ret = -1;
            assert(0 && "Unknown type.");
            break;
    }

    //printf("xtc_type = %s, return type: %s\n", xtc_type_name, h5_type_name);
    return ret;
}

static herr_t 
H5VL_xtc_dataset_get(void *dset, H5VL_dataset_get_t get_type,
    hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    //assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)dset;

    herr_t ret_value;
    xtc_object* obj = o->xtc_obj;

    switch(get_type){
        case H5VL_DATASET_GET_DAPL:                  /* access property list                */
            ;
            DEBUG_PRINT
            hid_t* ret_dapl = va_arg(arguments, hid_t*);
            *ret_dapl = H5P_DEFAULT;
            break;

        case H5VL_DATASET_GET_DCPL:                  /* creation property list              */
            ;
            DEBUG_PRINT
            hid_t dcpl = H5Pcreate (H5P_DATASET_CREATE);
            hid_t* ret_dcpl = va_arg(arguments, hid_t*);
            *ret_dcpl = dcpl;
            break;

        case H5VL_DATASET_GET_OFFSET:                /* offset                              */
            printf("%s:%d: H5VL_DATASET_GET_OFFSET\n", __func__, __LINE__);
            assert(0 && "DS not implemented: get offset");
            break;

        case H5VL_DATASET_GET_SPACE:                /* dataspace                           */
            assert(obj->ds_info);
            DEBUG_PRINT
            hid_t sid = H5Screate_simple(obj->ds_info->dim_cnt,
                    obj->ds_info->current_dims, obj->ds_info->maximum_dims);
            hid_t* ret_sid = va_arg(arguments, hid_t*);
            *ret_sid = sid;
            break;

        case H5VL_DATASET_GET_SPACE_STATUS:          /* space status                        */
            printf("%s:%d: H5VL_DATASET_GET_SPACE_STATUS\n", __func__, __LINE__);
            assert(0 && "DS not implemented: get space status");
            break;

        case H5VL_DATASET_GET_STORAGE_SIZE:
            ;
            DEBUG_PRINT
            int s_size = obj->ds_info->element_cnt * obj->ds_info->element_size;
            hsize_t* size = va_arg(arguments, hsize_t*);
            *size = s_size;
            DEBUG_PRINT
            break;

        case H5VL_DATASET_GET_TYPE:
            ;
            hid_t type = type_convert(obj->ds_info->type);
            hid_t* ret_tid = va_arg(arguments, hid_t*);
            *ret_tid = type;
            DEBUG_PRINT
            break;

        default:
            printf("%s:%d: Unknown get_type = %d\n", __func__, __LINE__, get_type);
            break;
    }
    //get_dcpl
    //get_dataspace

    /* sys metadata for ds
     *     H5Dget_space

        Data:
        H5Dread

        Metadata:
        H5Aiterate2: no attributes, 0.
        H5Dget_create_plist: copy the default
        H5Dget_type
        H5Dget_storage_size:  n * sizeof(type)

        Object:
        H5Oget_comment: no comment.
    */

    return ret_value;
} /* end H5VL_xtc_dataset_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_dataset_specific
 *
 * Purpose:     Specific operation on a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_dataset_specific(void *obj, H5VL_dataset_specific_t specific_type,
    hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_dataset_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_dataset_optional
 *
 * Purpose:     Perform a connector-specific operation on a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_dataset_optional(void *obj, H5VL_dataset_optional_t opt_type,
        hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL DATASET Optional\n");
//#endif
//
//    ret_value = H5VLdataset_optional(o->under_object, o->under_vol_id, dxpl_id, req, arguments);
//
//    /* Check for async request */
//    if(req && *req)
//        *req = H5VL_xtc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_xtc_dataset_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_dataset_close
 *
 * Purpose:     Closes a dataset.
 *
 * Return:      Success:    0
 *              Failure:    -1, dataset not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_dataset_close(void *dset, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    //assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)dset;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL DATASET Close\n");
#endif
//
//    ret_value = H5VLdataset_close(o->under_object, o->under_vol_id, dxpl_id, req);
//
//    /* Check for async request */
//    if(req && *req)
//        *req = H5VL_xtc_new_obj(*req, o->under_vol_id);
//
//    /* Release our wrapper, if underlying dataset was closed */
//    if(ret_value >= 0)
//        H5VL_xtc_free_obj(o);

    return ret_value;
} /* end H5VL_xtc_dataset_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_datatype_commit
 *
 * Purpose:     Commits a datatype inside a container.
 *
 * Return:      Success:    Pointer to datatype object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_datatype_commit(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id, hid_t tapl_id,
    hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *dt;
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    void *under;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL DATATYPE Commit\n");
//#endif
//
//    under = H5VLdatatype_commit(o->under_object, loc_params, o->under_vol_id, name, type_id, lcpl_id, tcpl_id, tapl_id, dxpl_id, req);
//    if(under) {
//        dt = H5VL_xtc_new_obj(under, o->under_vol_id);
//
//        /* Check for async request */
//        if(req && *req)
//            *req = H5VL_xtc_new_obj(*req, o->under_vol_id);
//    } /* end if */
//    else
//        dt = NULL;
//    return (void *)dt;
    return NULL;
} /* end H5VL_xtc_datatype_commit() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_datatype_open
 *
 * Purpose:     Opens a named datatype inside a container.
 *
 * Return:      Success:    Pointer to datatype object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_datatype_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t tapl_id, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *dt;
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    void *under;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL DATATYPE Open\n");
//#endif
//
//    under = H5VLdatatype_open(o->under_object, loc_params, o->under_vol_id, name, tapl_id, dxpl_id, req);
//    if(under) {
//        dt = H5VL_xtc_new_obj(under, o->under_vol_id);
//
//        /* Check for async request */
//        if(req && *req)
//            *req = H5VL_xtc_new_obj(*req, o->under_vol_id);
//    } /* end if */
//    else
//        dt = NULL;

    return (void *)dt;
} /* end H5VL_xtc_datatype_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_datatype_get
 *
 * Purpose:     Get information about a datatype
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_datatype_get(void *dt, H5VL_datatype_get_t get_type,
    hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)dt;
    herr_t ret_value;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL DATATYPE Get\n");
//#endif
//
//    ret_value = H5VLdatatype_get(o->under_object, o->under_vol_id, get_type, dxpl_id, req, arguments);
//
//    /* Check for async request */
//    if(req && *req)
//        *req = H5VL_xtc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_xtc_datatype_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_datatype_specific
 *
 * Purpose:     Specific operations for datatypes
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_datatype_specific(void *obj, H5VL_datatype_specific_t specific_type,
    hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL DATATYPE Specific\n");
#endif

//    // Save copy of underlying VOL connector ID and prov helper, in case of
//    // refresh destroying the current object
//    under_vol_id = o->under_vol_id;
//
//    ret_value = H5VLdatatype_specific(o->under_object, o->under_vol_id, specific_type, dxpl_id, req, arguments);
//
//    /* Check for async request */
//    if(req && *req)
//        *req = H5VL_xtc_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_xtc_datatype_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_datatype_optional
 *
 * Purpose:     Perform a connector-specific operation on a datatype
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_datatype_optional(void *obj, H5VL_datatype_optional_t opt_type,
        hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL DATATYPE Optional\n");
//#endif
//
//    ret_value = H5VLdatatype_optional(o->under_object, o->under_vol_id, dxpl_id, req, arguments);
//
//    /* Check for async request */
//    if(req && *req)
//        *req = H5VL_xtc_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_xtc_datatype_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_datatype_close
 *
 * Purpose:     Closes a datatype.
 *
 * Return:      Success:    0
 *              Failure:    -1, datatype not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_datatype_close(void *dt, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)dt;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL DATATYPE Close\n");
#endif

//    assert(o->under_object);
//
//    ret_value = H5VLdatatype_close(o->under_object, o->under_vol_id, dxpl_id, req);
//
//    /* Check for async request */
//    if(req && *req)
//        *req = H5VL_xtc_new_obj(*req, o->under_vol_id);
//
//    /* Release our wrapper, if underlying datatype was closed */
//    if(ret_value >= 0)
//        H5VL_xtc_free_obj(o);

    return ret_value;
} /* end H5VL_xtc_datatype_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_file_create
 *
 * Purpose:     Creates a container using this connector
 *
 * Return:      Success:    Pointer to a file object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_file_create(const char *name, unsigned flags, hid_t fcpl_id,
    hid_t fapl_id, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_info_t *info;
    H5VL_xtc_t *file;
    hid_t under_fapl_id;
    void *under;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL FILE Create\n");
//#endif
//
//    /* Get copy of our VOL info from FAPL */
//    H5Pget_vol_info(fapl_id, (void **)&info);
//
//    /* Copy the FAPL */
//    under_fapl_id = H5Pcopy(fapl_id);
//
//    /* Set the VOL ID and info for the underlying FAPL */
//    H5Pset_vol(under_fapl_id, info->under_vol_id, info->under_vol_info);
//
//    /* Open the file with the underlying VOL connector */
//    under = H5VLfile_create(name, flags, fcpl_id, under_fapl_id, dxpl_id, req);
//    if(under) {
//        file = H5VL_xtc_new_obj(under, info->under_vol_id);
//
//        /* Check for async request */
//        if(req && *req)
//            *req = H5VL_xtc_new_obj(*req, info->under_vol_id);
//    } /* end if */
//    else
//        file = NULL;
//
//    /* Close underlying FAPL */
//    H5Pclose(under_fapl_id);
//
//    /* Release copy of our VOL info */
//    H5VL_xtc_info_free(info);

    return (void *)file;
} /* end H5VL_xtc_file_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_file_open
 *
 * Purpose:     Opens a container created with this connector
 *
 * Return:      Success:    Pointer to a file object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_file_open(const char *name, unsigned flags, hid_t fapl_id,
    hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL FILE Open\n");
#endif

    /* Open the file with the underlying VOL connector */
    xtc_object* head_obj = xtc_file_open(name);
    if(head_obj->ref_cnt != 0){
        printf("Calling xtc_file_open() failed, ref_cnt checking error!\n");
        return NULL;
    }

    H5VL_xtc_t *file = H5VL_xtc_new_obj(head_obj);
    file->xtc_obj_type = XTC_FILE;
    file->obj_path = strdup("/"); //assume all files are root groups
    //H5VL_xtc_info_free(info);
    DEBUG_PRINT
    return (void *)file;
} /* end H5VL_xtc_file_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_file_get
 *
 * Purpose:     Get info about a file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_file_get(void *file, H5VL_file_get_t get_type, hid_t dxpl_id,
    void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)file;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_file_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_file_specific_reissue
 *
 * Purpose:     Re-wrap vararg arguments into a va_list and reissue the
 *              file specific callback to the underlying VOL connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_file_specific_reissue(void *obj, hid_t connector_id,
    H5VL_file_specific_t specific_type, hid_t dxpl_id, void **req, ...)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    va_list arguments;
    herr_t ret_value;

    va_start(arguments, req);
    ret_value = H5VLfile_specific(obj, connector_id, specific_type, dxpl_id, req, arguments);
    va_end(arguments);

    return ret_value;
} /* end H5VL_xtc_file_specific_reissue() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_file_specific
 *
 * Purpose:     Specific operation on file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_file_specific(void *file, H5VL_file_specific_t specific_type,
    hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)file;
    hid_t under_vol_id = -1;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_file_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_file_optional
 *
 * Purpose:     Perform a connector-specific operation on a file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_file_optional(void *obj, H5VL_file_optional_t opt_type,
        hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL File Optional\n");
#endif

    return ret_value;
} /* end H5VL_xtc_file_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_file_close
 *
 * Purpose:     Closes a file.
 *
 * Return:      Success:    0
 *              Failure:    -1, file not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_file_close(void *file, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    //assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)file;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL FILE Close\n");
#endif
    xtc_object* head = o->xtc_obj;
    //H5VL_xtc_free_obj(o);
    xtc_file_close(head);


//    ret_value = H5VLfile_close(o->under_object, o->under_vol_id, dxpl_id, req);
//
//    /* Check for async request */
//    if(req && *req)
//        *req = H5VL_xtc_new_obj(*req, o->under_vol_id);
//
//    /* Release our wrapper, if underlying file was closed */
//    if(ret_value >= 0)
//        H5VL_xtc_free_obj(o);

    //if(0 == xtc_obj->ref_cnt){

    //}
    return ret_value;
} /* end H5VL_xtc_file_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_group_create
 *
 * Purpose:     Creates a group inside a container
 *
 * Return:      Success:    Pointer to a group object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_group_create(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id,
    hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *group;
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    void *under;

    return (void *)group;
} /* end H5VL_xtc_group_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_group_open
 *
 * Purpose:     Opens a group inside a container
 *
 * Return:      Success:    Pointer to a group object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_group_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t gapl_id, hid_t dxpl_id, void **req)
{
    //assert(0);
    DEBUG_PRINT
    H5VL_xtc_t* target = _object_lookup(obj, loc_params);
    xtc_object* group = target->xtc_obj;

    H5VL_xtc_t* group_rt = H5VL_xtc_new_obj(group);//calloc(1, sizeof(H5VL_xtc_t));
    group_rt->xtc_obj = group;
    DEBUG_PRINT

    return (void *)group_rt;
} /* end H5VL_xtc_group_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_group_get
 *
 * Purpose:     Get info about a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_group_get(void *obj, H5VL_group_get_t get_type, hid_t dxpl_id,
    void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;
    return ret_value;
} /* end H5VL_xtc_group_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_group_specific
 *
 * Purpose:     Specific operation on a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_group_specific(void *obj, H5VL_group_specific_t specific_type,
    hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_group_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_group_optional
 *
 * Purpose:     Perform a connector-specific operation on a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_group_optional(void *obj, H5VL_group_optional_t opt_type,
        hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_group_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_group_close
 *
 * Purpose:     Closes a group.
 *
 * Return:      Success:    0
 *              Failure:    -1, group not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_group_close(void *grp, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    //assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)grp;
    herr_t ret_value = 0;

    return ret_value;
} /* end H5VL_xtc_group_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_link_create_reissue
 *
 * Purpose:     Re-wrap vararg arguments into a va_list and reissue the
 *              link create callback to the underlying VOL connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_link_create_reissue(H5VL_link_create_type_t create_type,
    void *obj, const H5VL_loc_params_t *loc_params, hid_t connector_id,
    hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req, ...)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    va_list arguments;
    herr_t ret_value;


    H5VL_xtc_t* target = _object_lookup(obj, loc_params);
    xtc_object* xtc_obj = target->xtc_obj;


    va_start(arguments, req);
    ret_value = H5VLlink_create(create_type, obj, loc_params, connector_id, lcpl_id, lapl_id, dxpl_id, req, arguments);
    va_end(arguments);

    return ret_value;
} /* end H5VL_xtc_link_create_reissue() */

/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_link_create
 *
 * Purpose:     Creates a hard / soft / UD / external link.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_link_create(H5VL_link_create_type_t create_type, void *obj,
    const H5VL_loc_params_t *loc_params, hid_t lcpl_id, hid_t lapl_id,
    hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_link_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_link_copy
 *
 * Purpose:     Renames an object within an HDF5 container and copies it to a new
 *              group.  The original name SRC is unlinked from the group graph
 *              and then inserted with the new name DST (which can specify a
 *              new path for the object) as an atomic operation. The names
 *              are interpreted relative to SRC_LOC_ID and
 *              DST_LOC_ID, which are either file IDs or group ID.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_link_copy(void *src_obj, const H5VL_loc_params_t *loc_params1,
    void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id,
    hid_t lapl_id, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o_src = (H5VL_xtc_t *)src_obj;
    H5VL_xtc_t *o_dst = (H5VL_xtc_t *)dst_obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_link_copy() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_link_move
 *
 * Purpose:     Moves a link within an HDF5 file to a new group.  The original
 *              name SRC is unlinked from the group graph
 *              and then inserted with the new name DST (which can specify a
 *              new path for the object) as an atomic operation. The names
 *              are interpreted relative to SRC_LOC_ID and
 *              DST_LOC_ID, which are either file IDs or group ID.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_link_move(void *src_obj, const H5VL_loc_params_t *loc_params1,
    void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id,
    hid_t lapl_id, hid_t dxpl_id, void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o_src = (H5VL_xtc_t *)src_obj;
    H5VL_xtc_t *o_dst = (H5VL_xtc_t *)dst_obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_link_move() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_link_get
 *
 * Purpose:     Get info about a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_link_get(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_link_get_t get_type, hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

    H5VL_xtc_t* target = _object_lookup(o, loc_params);
    xtc_object* xtc_obj = target->xtc_obj;
            
    return ret_value;
} /* end H5VL_xtc_link_get() */

/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_link_specific
 *
 * Purpose:     Specific operation on a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
int loop_children(bool recursive, xtc_object* xtc_obj, H5L_iterate2_t* op, void* op_data){
    if(!xtc_obj){
        printf("%s: null xtc_obj. return.\n", __func__);
        return 0;
    }

    int n_children = 0;
    xtc_object** children = xtc_get_children_list(xtc_obj, &n_children);
    hid_t gid = H5VLwrap_register(xtc_obj, H5I_GROUP);//obj

    if(n_children == 0 || !children){
        H5Idec_ref(gid);
        return H5_ITER_CONT;
    }

    assert(children && *children);

    for(int i = 0; i < n_children; i++){// iterate xtc_obj group
        char* link_name = children[i]->obj_path_abs;
        H5L_info2_t linfo;
        linfo.type = H5L_TYPE_HARD;
        linfo.corder_valid = false;
        linfo.corder = 0;
        linfo.cset = 0; //US ASCII
        linfo.u.token = *(H5O_token_t*)(children[i]->obj_token);

        int op_ret =  (*op)(gid, link_name, &linfo, op_data);//list metadata of children[i]

        if(op_ret != H5_ITER_CONT){
            H5Idec_ref(gid);
            return op_ret;
        }

        if(recursive && children[i]->obj_type == XTC_GROUP){
            op_ret = loop_children(recursive, children[i], op, op_data);
            if(op_ret != H5_ITER_CONT){
                H5Idec_ref(gid);
                return op_ret;
            }
        }
    }//end for(children)

    H5Idec_ref(gid);

    return H5_ITER_CONT;
}

static herr_t 
H5VL_xtc_link_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_link_specific_t specific_type, hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    //assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;

    herr_t ret_value;

    H5VL_xtc_t* target = _object_lookup(o, loc_params);
    assert(target);
    xtc_object* target_obj = target->xtc_obj;

    switch(specific_type){
        case H5VL_LINK_DELETE:
            printf("specific_type = H5VL_LINK_DELETE \n");
            assert(0 && "Not implemented");
            break;
        case H5VL_LINK_EXISTS:
            printf("specific_type = H5VL_LINK_EXISTS \n");
            assert(0 && "Not implemented");
            break;
        case H5VL_LINK_ITER: {
            hbool_t recursive = va_arg(arguments, hbool_t);
            H5_index_t idx_type = va_arg(arguments, H5_index_t);
            H5_iter_order_t order = va_arg(arguments, H5_iter_order_t);
            hsize_t* idx_p = va_arg(arguments, hsize_t*);
            H5L_iterate2_t op = va_arg(arguments, H5L_iterate2_t);
            void* op_data = va_arg(arguments, void*);
            if(target->xtc_obj->obj_type == XTC_GROUP || target->xtc_obj->obj_type ==XTC_HEAD){
                ret_value = loop_children(recursive, target->xtc_obj, &op, op_data);
            } else {
                assert(0 && "DO NOT loop through a non-group link!");
            }

        }
            break;
        default:
            assert(0 && "Not Implemented");
            break;
    }//switch(specific_type)
    return ret_value;
} /* end H5VL_xtc_link_specific() */

/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_link_optional
 *
 * Purpose:     Perform a connector-specific operation on a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_xtc_link_optional(void *obj, H5VL_link_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_link_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_object_open
 *
 * Purpose:     Opens an object inside a container.
 *
 * Return:      Success:    Pointer to object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_xtc_object_open(void *obj, const H5VL_loc_params_t *loc_params,
    H5I_type_t *opened_type, hid_t dxpl_id, void **req)
{
    //printf("\n");
    DEBUG_PRINT

    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;

    H5VL_xtc_t* target = _object_lookup(o, loc_params);
    assert(target);
    switch(target->xtc_obj_type){
        case XTC_FILE:
            *opened_type = H5I_FILE;
            break;
        case XTC_HEAD:
        case XTC_GROUP:
            *opened_type = H5I_GROUP;
            break;
        case XTC_TIME_DS:
        case XTC_DS:
            *opened_type = H5I_DATASET;
            break;
        default:
            printf("%s: type = Unknown o->xtc_obj_type : %d, not implemented. \n", __func__, target->xtc_obj_type);
            break;
    }

//    switch(*opened_type){
//        case H5I_FILE:
//            //printf("%s: opened_type = H5I_FILE: %d\n", __func__, *opened_type);
//            break;
//        case H5I_GROUP:
//            //printf("%s: opened_type = H5I_GROUP: %d \n", __func__, *opened_type);
//            break;
//        case H5I_DATATYPE:
//            //printf("%s: opened_type = H5I_DATATYPE: %d.\n", __func__, *opened_type);
//            break;
//        case H5I_DATASPACE:
//            //printf("%s: opened_type = H5I_DATASPACE: %d, not implemented. \n", __func__, *opened_type);
//            break;
//        case H5I_DATASET:
//            //printf("%s: opened_type = H5I_DATASET: %d, not implemented. \n", __func__, *opened_type);
//            break;
//        default:
//            //printf("%s: type = Unknown opened_type type: %d \n", __func__, *opened_type);
//            break;
//    }

    return (void *)target;
} /* end H5VL_xtc_object_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_object_copy
 *
 * Purpose:     Copies an object inside a container.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_object_copy(void *src_obj, const H5VL_loc_params_t *src_loc_params,
    const char *src_name, void *dst_obj, const H5VL_loc_params_t *dst_loc_params,
    const char *dst_name, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id,
    void **req)
{
    DEBUG_PRINT
    assert(0 && "breakpoint");
    H5VL_xtc_t *o_src = (H5VL_xtc_t *)src_obj;
    H5VL_xtc_t *o_dst = (H5VL_xtc_t *)dst_obj;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_object_copy() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_object_get
 *
 * Purpose:     Get info about an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */

int _xtc_object_get_info(H5VL_xtc_t* xtc_obj, H5O_info2_t* oinfo_out, unsigned fields){
    DEBUG_PRINT
    assert(xtc_obj && oinfo_out);
    if(fields & H5O_INFO_BASIC){//meaning fields is H5O_INFO_BASIC
        switch(xtc_obj->xtc_obj_type){
            case XTC_FILE:
                oinfo_out->type = H5O_TYPE_GROUP;
                break;
            case XTC_HEAD:
            case XTC_GROUP:
                oinfo_out->type = H5O_TYPE_GROUP;
                break;
            case XTC_TIME_DS:
            case XTC_DS:
                oinfo_out->type = H5O_TYPE_DATASET;
                break;
            default:
                printf("Unknown xtc_obj type: %d\n", xtc_obj->xtc_obj_type);
                assert(0);
                break;
        }
        oinfo_out->fileno = xtc_obj->xtc_obj->fd;//fd
        oinfo_out->rc = 1;//keep it to 1.
        assert(xtc_obj->xtc_obj->obj_token);
        oinfo_out->token = *(H5O_token_t*)(xtc_obj->xtc_obj->obj_token);// Unique ID for xtc object in current runtime scope.
    }

    if(fields & H5O_INFO_TIME){
        oinfo_out->atime = 0;
        oinfo_out->btime = 0;
        oinfo_out->ctime = 0;
        oinfo_out->mtime = 0;
    }

    if(fields & H5O_INFO_NUM_ATTRS){
        oinfo_out->num_attrs = 0;
    }
    return 0;
}

static herr_t
H5VL_xtc_object_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_get_t get_type,
        hid_t dxpl_id, void **req, va_list arguments)
{
    //assert(0 && "breakpoint");
    DEBUG_PRINT

    herr_t ret_value;
    H5VL_xtc_t* input_obj = NULL;

    switch(loc_params->obj_type){
        case H5I_FILE:
            DEBUG_PRINT
            input_obj = (H5VL_xtc_t *)obj;//H5VL_xtc_new_obj(h5vl_file_obj->xtc_obj); //or obj itself IS this input_obj?
            break;
        case H5I_GROUP:
            DEBUG_PRINT
            //Parameter obj is the xtc_obj used to generate gid and wrap to a vol_obj, and xtc_obj is the vol_obj->data
            input_obj = H5VL_xtc_new_obj((xtc_object*)obj);
            break;
        case H5I_DATASET:
            DEBUG_PRINT
            assert(0 && "DS not implemented.");
            //input_obj = (H5VL_xtc_t *)obj;
            input_obj = H5VL_xtc_new_obj((xtc_object*)obj);
            break;
        default:
            printf("%s: Unsupported type: %d \n", __func__, loc_params->obj_type);
            assert(0 && "Unsupported type not implemented.");
            break;
    }
    H5VL_xtc_t* target = _object_lookup(input_obj, loc_params);
    assert(target);
    switch(get_type){
        case H5VL_OBJECT_GET_FILE:
            printf("%s: type = H5VL_OBJECT_GET_FILE\n", __func__);
            assert(0 && "Not implemented");
            break;
        case H5VL_OBJECT_GET_NAME:
            printf("%s: type = H5VL_OBJECT_GET_NAME\n", __func__);
            assert(0 && "Not implemented");
            break;
        case H5VL_OBJECT_GET_TYPE:
            printf("%s: type = H5VL_OBJECT_GET_TYPE\n", __func__);
            assert(0 && "Not implemented");
            break;
        case H5VL_OBJECT_GET_INFO:
            ;
            H5O_info2_t* target_oinfo = va_arg(arguments, H5O_info2_t*);//get param FROM va_list
            unsigned fields = va_arg(arguments, unsigned);

            _xtc_object_get_info(target, target_oinfo, fields);  //
            break;
        default:
            printf("%s: type = Unknown get type: %d \n", __func__, get_type);
            assert(0 && "Not implemented");
            break;
    }
    DEBUG_PRINT
    return ret_value;
} /* end H5VL_xtc_object_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_object_specific
 *
 * Purpose:     Specific operation on an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_object_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_object_specific_t specific_type, hid_t dxpl_id, void **req,
    va_list arguments)
{
    assert(0 && "breakpoint");
    DEBUG_PRINT
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

    H5VL_xtc_t* target = _object_lookup(o, loc_params);
    xtc_object* xtc_obj = target->xtc_obj;

    switch(specific_type){
        case H5VL_OBJECT_LOOKUP:
        case H5VL_OBJECT_VISIT:
        case H5VL_OBJECT_EXISTS:
        default:
            printf("%s:%d:  specific_type = %d\n", __func__, __LINE__, specific_type);
            assert(0 && "Not implemented");
            break;
    }

    return ret_value;
} /* end H5VL_xtc_object_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_object_optional
 *
 * Purpose:     Perform a connector-specific operation for an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_object_optional(void *obj, H5VL_object_optional_t opt_type, hid_t dxpl_id, void **req, va_list arguments)
{
    //assert(0 && "breakpoint");
    DEBUG_PRINT
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

    return ret_value;
} /* end H5VL_xtc_object_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_request_wait
 *
 * Purpose:     Wait (with a timeout) for an async operation to complete
 *
 * Note:        Releases the request if the operation has completed and the
 *              connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_request_wait(void *obj, uint64_t timeout,
    H5ES_status_t *status)
{
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL REQUEST Wait\n");
//#endif
//
//    ret_value = H5VLrequest_wait(o->under_object, o->under_vol_id, timeout, status);
//
//    if(ret_value >= 0 && *status != H5ES_STATUS_IN_PROGRESS)
//        H5VL_xtc_free_obj(o);

    return ret_value;
} /* end H5VL_xtc_request_wait() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_request_notify
 *
 * Purpose:     Registers a user callback to be invoked when an asynchronous
 *              operation completes
 *
 * Note:        Releases the request, if connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_request_notify(void *obj, H5VL_request_notify_t cb, void *ctx)
{
    assert(0 && "breakpoint");
    DEBUG_PRINT
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL REQUEST Wait\n");
//#endif
//
//    ret_value = H5VLrequest_notify(o->under_object, o->under_vol_id, cb, ctx);
//
//    if(ret_value >= 0)
//        H5VL_xtc_free_obj(o);

    return ret_value;
} /* end H5VL_xtc_request_notify() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_request_cancel
 *
 * Purpose:     Cancels an asynchronous operation
 *
 * Note:        Releases the request, if connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_request_cancel(void *obj)
{
    assert(0 && "breakpoint");
    DEBUG_PRINT
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL REQUEST Cancel\n");
//#endif
//
//    ret_value = H5VLrequest_cancel(o->under_object, o->under_vol_id);
//
//    if(ret_value >= 0)
//        H5VL_xtc_free_obj(o);

    return ret_value;
} /* end H5VL_xtc_request_cancel() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_request_specific_reissue
 *
 * Purpose:     Re-wrap vararg arguments into a va_list and reissue the
 *              request specific callback to the underlying VOL connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_request_specific_reissue(void *obj, hid_t connector_id,
    H5VL_request_specific_t specific_type, ...)
{
    assert(0 && "breakpoint");
    DEBUG_PRINT
    va_list arguments;
    herr_t ret_value;

    va_start(arguments, specific_type);
    ret_value = H5VLrequest_specific(obj, connector_id, specific_type, arguments);
    va_end(arguments);

    return ret_value;
} /* end H5VL_xtc_request_specific_reissue() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_request_specific
 *
 * Purpose:     Specific operation on a request
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_request_specific(void *obj, H5VL_request_specific_t specific_type,
    va_list arguments)
{
    assert(0 && "breakpoint");
    DEBUG_PRINT
    herr_t ret_value = -1;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL REQUEST Specific\n");
#endif
    return ret_value;
} /* end H5VL_xtc_request_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_request_optional
 *
 * Purpose:     Perform a connector-specific operation for a request
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_request_optional(void *obj, va_list arguments)
{
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL REQUEST Optional\n");
#endif
//
//    ret_value = H5VLrequest_optional(o->under_object, o->under_vol_id, arguments);

    return ret_value;
} /* end H5VL_xtc_request_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_request_free
 *
 * Purpose:     Releases a request, allowing the operation to complete without
 *              application tracking
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t 
H5VL_xtc_request_free(void *obj)
{
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL REQUEST Free\n");
#endif
//
//    ret_value = H5VLrequest_free(o->under_object, o->under_vol_id);
//
//    if(ret_value >= 0)
//        H5VL_xtc_free_obj(o);

    return ret_value;
} /* end H5VL_xtc_request_free() */



static herr_t H5VL_xtc_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl, const struct H5VL_class_t **conn_cls){
    /* Sanity check */
    assert(0 && "breakpoint");
    assert(conn_cls);

    /* Retrieve the native VOL connector class */
    *conn_cls = &H5VL_xtc_g;
    return 0;
}

static herr_t H5VL_xtc_introspect_opt_query(void *obj, H5VL_subclass_t cls, int opt_type, hbool_t *supported){
    //assert(0 && "breakpoint");
    assert(supported);

    *supported = false;
    return 0;
}
/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_blob_put
 *
 * Purpose:     Handles the blob 'put' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_xtc_blob_put(void *obj, const void *buf, size_t size,
    void *blob_id, void *ctx)
{
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL BLOB Put\n");
#endif
//
//    ret_value = H5VLblob_put(o->under_object, o->under_vol_id, buf, size,
//        blob_id, ctx);

    return ret_value;
} /* end H5VL_xtc_blob_put() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_blob_get
 *
 * Purpose:     Handles the blob 'get' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_xtc_blob_get(void *obj, const void *blob_id, void *buf, size_t size, void *ctx)
{
    assert(0 && "breakpoint");
    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_XTC_LOGGING
    printf("------- XTC VOL BLOB Get\n");
#endif

//    ret_value = H5VLblob_get(o->under_object, o->under_vol_id, blob_id, buf,
//        size, ctx);

    return ret_value;
} /* end H5VL_xtc_blob_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_xtc_blob_specific
 *
 * Purpose:     Handles the blob 'specific' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
//herr_t
//H5VL_xtc_blob_specific(void *obj, void *blob_id,
//    H5VL_blob_specific_t specific_type, va_list arguments)
//{
//    H5VL_xtc_t *o = (H5VL_xtc_t *)obj;
//    herr_t ret_value;
//
//#ifdef ENABLE_XTC_LOGGING
//    printf("------- XTC VOL BLOB Specific\n");
//#endif
//
////    ret_value = H5VLblob_specific(o->under_object, o->under_vol_id, blob_id,
////        specific_type, arguments);
//
//    return ret_value;
//} /* end H5VL_xtc_blob_specific() */

herr_t H5VL_xtc_token_cmp(void *obj, const H5O_token_t *token1, const H5O_token_t *token2, int *cmp_value){
    assert(0 && "breakpoint");
    return 0;
}

herr_t H5VL_xtc_token_to_str(void *obj, H5I_type_t obj_type, const H5O_token_t *token, char **token_str){
    //assert(0 && "breakpoint");
    *token_str = xtc_h5token_to_str(token);
    return 0;
}

herr_t H5VL_xtc_token_from_str(void *obj, H5I_type_t obj_type, const char *token_str, H5O_token_t *token){
    assert(0 && "breakpoint");
    return 0;
}

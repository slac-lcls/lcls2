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

/*
 * Purpose:	The public header file for the XTC VOL connector.
 */

#ifndef _H5VLxtc_H
#define _H5VLxtc_H

/* Public headers needed by this file */
#include "H5VLpublic.h"        /* Virtual Object Layer                 */
#include "xtc_io_api_c.h"
/* Identifier for the xtc VOL connector */
#define XTC_VOL	(H5VL_xtc_register())

/* Characteristics of the xtc VOL connector */
#define XTC_VOL_NAME        "xtc_vol"
#define XTC_VOL_VALUE       1003           /* VOL connector ID */
#define XTC_VOL_VERSION     0

/* XTC VOL connector info */
//This is a terminal VOL: no under_vol, under_* members are place holders to satisfy compiler.
typedef struct H5VL_xtc_info_t {
    hid_t under_vol_id;         /* VOL ID for under VOL */
    void *under_vol_info;       /* VOL info for under VOL */
} H5VL_xtc_info_t;

extern const H5VL_class_t H5VL_xtc_g;
//One helper per file.

#ifdef __cplusplus
extern "C" {
#endif

H5_DLL hid_t H5VL_xtc_register(void);

#ifdef __cplusplus
}
#endif

#endif /* _H5VLxtc_H */


/*
 * prov_test.c
 *
 *  Created on: Aug 19, 2019
 *      Author: tonglin
 */

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <mpi.h>
#include <hdf5.h>
#include "xtc_vol.h"
#include "util_debug.h"

int my_rank;
int comm_size;
extern int MY_RANK_DEBUG;
unsigned long public_get_time_stamp_us(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

unsigned long ds_test(int benchmark_type, const char* file_name, hid_t fapl, int num_ops){
    DEBUG_PRINT
    hid_t dataset_id, dataspace_id;
    herr_t status;
    char ds_name[32] = "";
    hsize_t     dims[2];
    int         i, j;
    int dset_data[4][6];
    /* Initialize the dataset. */
    for (i = 0; i < 4; i++)
       for (j = 0; j < 6; j++)
          dset_data[i][j] = i * 6 + j + 1;

    hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);//H5P_DEFAULT

    /* Create the data space for the dataset. */
    dims[0] = 4;
    dims[1] = 6;
    dataspace_id = H5Screate_simple(2, dims, NULL);

    unsigned long t1 = public_get_time_stamp_us();
    if(benchmark_type == 0){//baseline
        for(i = 0; i < comm_size; i++) {
            for(j = 0; j < num_ops; j++) {
                sprintf(ds_name, "/dset_%d_%d", i, j);
                dataset_id = H5Dcreate2(file_id, ds_name, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
                status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
                status = H5Dclose(dataset_id);
            }
        }
    }
    unsigned long t2 = public_get_time_stamp_us();

    status = H5Sclose(dataspace_id);

    status = H5Fclose(file_id);

    /* Re-open the file in serial and verify contents created independently */
    file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);

    for(i = 0; i < comm_size; i++) {
        for(int j = 0; j < num_ops; j++){
            sprintf(ds_name, "/dset_%d_%d", i, j);
            dataset_id = H5Dopen2(file_id, ds_name, H5P_DEFAULT);
            status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
            status = H5Dclose(dataset_id);
            DEBUG_PRINT
        }
    }
    H5Fclose(file_id);
    return t2 - t1;
}

unsigned long group_test(int benchmark_type, const char* file_name, hid_t fapl, int num_ops){
    DEBUG_PRINT
    hid_t group_id;
    herr_t status;
    char group_name[32] = "";

    /* Create a new file using default properties. */
    hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    unsigned long t1 = public_get_time_stamp_us();
    if(benchmark_type == 0){//baseline
        for(int i = 0; i < comm_size; i++) {
            DEBUG_PRINT
            for(int j = 0; j < num_ops; j++){
                sprintf(group_name, "/group_%d_%d", i, j);
                group_id = H5Gcreate2(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                DEBUG_PRINT
                status = H5Gclose(group_id);
                DEBUG_PRINT
            }
        }
    }
    unsigned long t2 = public_get_time_stamp_us();
    H5Fclose(file_id);

    /* Re-open the file in serial and verify contents created independently */
    file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);

    for(int i = 0; i < comm_size; i++) {
        for(int j = 0; j < num_ops; j++){
            sprintf(group_name, "/group_%d_%d", i, j);
            group_id = H5Gopen2(file_id, group_name, H5P_DEFAULT);
            status = H5Gclose(group_id);
            DEBUG_PRINT
        }
    }
    H5Fclose(file_id);
    return t2 - t1;
}

int main(int argc, char* argv[])
{
    hid_t fapl;
    const char* file_name = "xtc_test.h5";

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    printf("HDF5 XTC VOL test start...pid = %d, rank = %d\n", getpid(), my_rank);
    MY_RANK_DEBUG = my_rank;
    int benchmark_type = 0;
    unsigned long time_window = 50;
    //printf("1\n");

    //printf("1.2, time_window = %lu\n", time_window);
    /* Create a new file using default properties. */

    fapl = H5Pcreate(H5P_FILE_ACCESS);
    //printf("1.3\n");

    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    //printf("1.4\n");


    {//XTC_VOL
        //hid_t xtc_vol_id = H5VLregister_connector_by_name("xtc_vol", H5P_DEFAULT);

        extern const H5VL_class_t H5VL_xtc_g;
       	hid_t xtc_vol_id = H5VLregister_connector(&H5VL_xtc_g, H5P_DEFAULT);
        printf("1.5, xtc_vol_id = %llx\n", xtc_vol_id);
        H5VL_xtc_info_t xtc_vol_info;

        //hid_t baseline_vol_id = H5VLregister_connector_by_value(0, H5P_DEFAULT);

        xtc_vol_info.under_vol_id = H5VLregister_connector_by_value(0, H5P_DEFAULT);
        // printf("1.52, xtc_vol_info.under_vol_id = %llx\n", xtc_vol_info.under_vol_id);

        xtc_vol_info.under_vol_info = NULL;
//        xtc_vol_info.mpi_comm = MPI_COMM_WORLD;
//        xtc_vol_info.mpi_info = MPI_INFO_NULL;
//        xtc_vol_info.time_window_size = time_window;
//        xtc_vol_info.mode = benchmark_type;
//        xtc_vol_info.world_size = comm_size;
//        xtc_vol_info.my_rank = my_rank;
        H5Pset_vol(fapl, xtc_vol_id, &xtc_vol_info);
        //printf("1.6\n");

        H5VLclose(xtc_vol_id);
    }

    int num_ops = 1;
    //========================  Sub Test cases  ======================
    unsigned long t;
    t = ds_test(benchmark_type, file_name, fapl, num_ops);
    printf("HDF5 XTC VOL test done. ds_test take %lu usec, avg = %lu\n", t, (t / num_ops));

    t = group_test(benchmark_type, file_name, fapl, num_ops);
    printf("HDF5 XTC VOL test done. group_test took %lu usec, avg = %lu\n", t, (t / num_ops));

    //=================================================================
    H5Pclose(fapl);

    H5close();
    //rintf("HDF5 library shut down.\n");

    DEBUG_PRINT

    MPI_Finalize();
    return 0;
}


#include <chrono>
#include <math.h>
#include <sys/stat.h>

#include "H5Cpp.h"
using namespace H5;

// Compile on psbuild-rhel7 with
// source /reg/g/psdm/etc/psconda.sh


long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
};


void loop_write(const char* filename, size_t loop_limit, hsize_t chunk_size, size_t num_bytes){
    const H5std_string FILE_NAME(filename);
    const H5std_string DATASETNAME("ExtendibleArray");

   std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    hsize_t ext_size = 1, offset[] = {0}, chunk_dims[1] = {chunk_size};
    const hsize_t maxdims[] = {H5S_UNLIMITED}, dimsext[] = {1};
    int vl_rank =1;      //variable length rank is 1
    hsize_t vl_size[1] = {ext_size};    // initial size of data

    // Declare data to append. It's full of garbage from memory
    int data_ext[num_bytes];

    // Variable length structure for data
    hvl_t dat[1];

    // Define a variable length datatype
    VarLenType int_var_type(&PredType::NATIVE_INT);

    // Create a new file using the default property lists.
    H5File file(FILE_NAME, H5F_ACC_TRUNC);

    // Create a dataspace
    DataSpace *dspace = new DataSpace(vl_rank, vl_size, maxdims);

    // Modify dataset creation property to enable chunking
    DSetCreatPropList prop;
    prop.setChunk(1, chunk_dims);

    // Create the dataset.
    DataSet *dataset = new DataSet(file.createDataSet(DATASETNAME, int_var_type, *dspace, prop));


    // Assign the data to the hvl_t object
    dat[0].len = sizeof(data_ext)/sizeof(data_ext[0]);
    dat[0].p = &data_ext[0];

    // Write data to the dataset.
    for(size_t i=0; i<loop_limit; i++){

        vl_size[0] += dimsext[0]*i;
        offset[0] = vl_size[0]-1;
        dataset->extend(vl_size);

        DataSpace *filespace = new DataSpace(dataset -> getSpace ());
        filespace->selectHyperslab(H5S_SELECT_SET, dimsext, offset);


        DataSpace *memspace = new DataSpace(1, dimsext, NULL);

        dataset->write(dat, int_var_type, *memspace, *filespace);
        delete filespace;
        delete memspace;
    };
    prop.close();
    delete dspace;
    delete dataset;
    file.close();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    auto fileSize = (float) GetFileSize(FILE_NAME)/1000000; //MB
    float av_speed = 1000*fileSize/(duration); // MB/s
    float av_freq = float(loop_limit)/duration;
    float hdf_ratio = 1000000*fileSize/(num_bytes*loop_limit*4.0);

    printf("%-20i%-20i%-20i%-20i%-20.2f%-20.2f%-20.2f%-20.2f\n", chunk_size , loop_limit, num_bytes, duration, fileSize, hdf_ratio, av_speed, av_freq); 

}



int main (int argc, char *argv[])
{
    printf("%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s\n", "Chunk size", "Loop limit", "Bytes/extension", "Duration (ms)", "Filesize (MB)", "HDF Ratio", "Write speed (MB/s)", "Frequency (kHz)");
     
    int loop_limit = 10000;
    int num_bytes = 1000000;
    // Increment the chunk size by powers of two up to the loop limit
    for(size_t i=0; pow(2,i)<loop_limit;i++){
        loop_write(argv[1], loop_limit,pow(2,i),num_bytes);
    }

    return 0;  // successfully terminated
}


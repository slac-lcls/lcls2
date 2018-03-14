#include <chrono>
#include <math.h>
#include <sys/stat.h>

#include "H5Cpp.h"
using namespace H5;

long GetFileSize(std::string filename)
{
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  return rc == 0 ? stat_buf.st_size : -1;
}

void loop_write(const char* filename, int loop_limit, hsize_t chunk_size, hsize_t num_bytes){
    const H5std_string FILE_NAME(filename);
    const H5std_string DATASETNAME("ExtendibleArray");

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    hsize_t dims[2] = {1,num_bytes};        // dataset dimensions at creation
    hsize_t maxdims[2] = {H5S_UNLIMITED, num_bytes}; 
    hsize_t chunk_dims[2] ={chunk_size, num_bytes};
    int num_ints = (int)num_bytes/4;
    int32_t  data[num_bytes];//= { {1, 1}};    // data to writ
    // Variables used in extending and writing to the extended portion of dataset 
    hsize_t size[2];
    hsize_t offset[2];
    hsize_t dimsext[2] = {1, num_bytes};         // extend dimensions 

    // Create a new file using the default property lists. 
    H5File file(FILE_NAME, H5F_ACC_TRUNC);

    DataSpace *dataspace = new DataSpace (2, dims, maxdims);

    // Modify dataset creation property to enable chunking
    DSetCreatPropList prop;
    prop.setChunk(2, chunk_dims);

    // Increase cache size to match chunks. Guessing on the parameters here. 
    size_t rd_chunk_bytes = chunk_dims[0]*8;
    FileAccPropList fprop;
    fprop.setCache((int) chunk_dims[0], (size_t) chunk_dims[0], rd_chunk_bytes, 1);


    // Create the chunked dataset.  Note the use of pointer.
    DataSet *dataset = new DataSet(file.createDataSet( DATASETNAME, 
                                                       PredType::STD_I32LE, *dataspace, prop) );
    for(int i=0; i<loop_limit; i++){
        // Extend the dataset. Dataset becomes n+1 x 3.
        size[0] = dims[0] + i*dimsext[0];
        size[1] = dims[1];
        dataset->extend(size); 

        // Select a hyperslab in extended portion of the dataset.
        DataSpace *filespace = new DataSpace(dataset->getSpace ());
        offset[0] = size[0]-1;
        offset[1] = 0;
        filespace->selectHyperslab(H5S_SELECT_SET, dimsext, offset);

        // Define memory space.
        DataSpace *memspace = new DataSpace(2, dimsext, NULL);
        // Write data to the extended portion of the dataset.
        dataset->write(data, PredType::NATIVE_INT, *memspace, *filespace);
        // delete filespace;
        delete filespace;
        delete memspace;

         };

    // Close all objects and file.
    prop.close();
    delete dataspace;
    delete dataset;
    file.close();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    auto fileSize = (float) GetFileSize(FILE_NAME)/1000000; //MB
    float av_speed = 1000*fileSize/(duration); // MB/s
    float av_freq = float(loop_limit)/duration;
    float hdf_ratio = 1000000*fileSize/(num_bytes*loop_limit*4.0);


    printf("%-20i%-20i%-20i%-20i%-20.2f%-20.2f%-20.2f%-20.2f\n", chunk_size , loop_limit, 4*num_bytes, duration, fileSize, hdf_ratio, av_speed, av_freq); 
        };

int main (int argc, char *argv[])
{

    const H5std_string FILE_NAME(argv[1]);   
    const H5std_string DATASETNAME("ExtendibleArray");


    // void loop_write(const char* filename, size_t loop_limit, hsize_t chunk_size, size_t num_bytes){
    auto chunk_size = (hsize_t) atoi(argv[3]);
    if(atoi(argv[3])==1){
        printf("%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s\n", "Chunk size", "Loop limit", "Bytes/extension", "Duration (ms)", "Filesize (MB)", "HDF Ratio", "Write speed (MB/s)", "Frequency (kHz)");
    };
    loop_write(argv[1], atoi(argv[2]), chunk_size,(size_t)atoi(argv[4])/4);
    return 0;  // successfully terminated
}

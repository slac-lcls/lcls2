#include <iostream>
using std::cout;
using std::endl;
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <math.h>
#include <sys/stat.h>

#include "H5Cpp.h"
using namespace H5;
using namespace std::chrono;

std::string filename = "/nvme1n1/hfdata.h5";
const H5std_string FILE_NAME(filename);
const H5std_string DATASETNAME("ExtendibleArray");

long GetFileSize(std::string filename)
{
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  return rc == 0 ? stat_buf.st_size : -1;
}

void loop_write(int loop_limit, hsize_t chunk_size, hsize_t num_bytes){
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    hsize_t dims[2] = {1,num_bytes};        // dataset dimensions at creation
    hsize_t maxdims[2] = {H5S_UNLIMITED, num_bytes}; 
    hsize_t chunk_dims[2] ={chunk_size, num_bytes};
    int32_t  data[1][num_bytes];//= { {1, 1}};    // data to write 

    // Variables used in extending and writing to the extended portion of dataset 

    hsize_t size[2];
    hsize_t offset[2];
    hsize_t dimsext[2] = {1, num_bytes};         // extend dimensions 
    //    int loop_limit = 10;

    // Create a new file using the default property lists. 
    H5File file(FILE_NAME, H5F_ACC_TRUNC);

    // Create the data space for the dataset.  Note the use of pointer
    // for the instance 'dataspace'.  It can be deleted and used again
    // later for another dataspace.  An HDF5 identifier can be closed
    // by the destructor or the method 'close()'.
    DataSpace *dataspace = new DataSpace (2, dims, maxdims);

    // Modify dataset creation property to enable chunking
    DSetCreatPropList prop;
    prop.setChunk(2, chunk_dims);

    // Create the chunked dataset.  Note the use of pointer.
    DataSet *dataset = new DataSet(file.createDataSet( DATASETNAME, 
                                                       PredType::STD_I32LE, *dataspace, prop) );
    //int i=0;
    // Write data to dataset.
    // dataset->write(data, PredType::NATIVE_INT);

    for(int i=0; i<loop_limit; i++){
        // Extend the dataset. Dataset becomes n+1 x 3.
        size[0] = dims[0] + i*dimsext[0];
        size[1] = dims[1];
        dataset->extend(size); 

        // Make some random numbers
        for(hsize_t j=0; i<dimsext[1]; i++){
        data[0][i] = rand();
        }
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

    size_t mem_size = dataset->getInMemDataSize();
    // Close all objects and file.
    delete dataset;



    delete dataspace;
    prop.close();
    file.close();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<milliseconds>( t2 - t1 ).count();
    long fileSize = GetFileSize(filename);
    float av_speed = (float)fileSize/(float)duration; // MB/s
    float av_freq = float(loop_limit)/duration;
    float hdf_ratio = float(fileSize)/(num_bytes*loop_limit*4.0);


    printf("\n");
    printf("Chunk size was %i\n", chunk_size);
    printf("Loop limit was %i\n", loop_limit);
    printf("Number of ints per extension was %i\n", num_bytes);
    printf("Duration was %.2d ms\n", duration);
    printf("Size of dataset is %.2f MB\n", (float)fileSize/1000000);
    printf("HDF5 ratio was %.2f\n", hdf_ratio);
    printf("Average speed was %.2f MB/s\n", av_speed/1000 );
    printf("Average frequency was %.2f kHz\n", av_freq);
    printf("\n");
}

int main (void)
{
    int loop_limit = 10000;

    for(auto i=0; pow(2,i)<loop_limit; i++){
    // loop limit, chunk size, integers per extension
        loop_write(loop_limit,pow(2,i),2);
    };
    return 0;  // successfully terminated
}

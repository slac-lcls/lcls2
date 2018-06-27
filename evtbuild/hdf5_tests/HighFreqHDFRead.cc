#include <chrono>
#include <math.h>
#include <sys/stat.h>

#include "H5Cpp.h"

using namespace H5;

const H5std_string DATASET_NAME( "ExtendibleArray" );
hsize_t dims[2]; 	// dataset dimensions

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
};

void loop_read(const char* filename, int buffer_size){
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    const H5std_string FILE_NAME( filename );
    FileCreatPropList fcparm;
    FileAccPropList fprop;

    //Set a 64 Mb cache
    fprop.setCache((int) 0, (size_t) buffer_size, buffer_size*1048576, 1);

    H5File file(FILE_NAME.c_str(), H5F_ACC_RDONLY,fcparm, fprop);
    DataSet dataset = file.openDataSet(DATASET_NAME.c_str());

    H5T_class_t type_class = dataset.getTypeClass();

    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentDims( dims );

    hsize_t row_sel[1] = {dims[1]}, offset[2] = {0, 0}, count[2] = {1, dims[1]};
    int row_out[dims[1]];

    hsize_t chunk_dims[2];
    int rank_chunk;
    DSetCreatPropList cparms = dataset.getCreatePlist();
    rank_chunk = cparms.getChunk( 2, chunk_dims);

    for(hsize_t i =0; i<dims[0]; i++){
        DataSpace memspace(1, row_sel);
        dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );
        dataset.read(row_out, PredType::NATIVE_INT, memspace, dataspace);
        offset[0]+=1;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    auto fileSize = (float) GetFileSize(FILE_NAME)/1000000; //MB
    float av_speed = 1000*fileSize/(duration); // MB/s
    float av_freq = float(dims[0])/duration;
    float hdf_ratio = 1000000*fileSize/(sizeof(row_out)*dims[0]);

    printf("%-20i%-20i%-20i%-20i%-20.2f%-20.2f%-20.2f%-20.2f\n", chunk_dims[0], dims[0], dims[1]*4, duration, fileSize, hdf_ratio, av_speed, av_freq); 
}


int main(int argc,  char *argv[]){
    // printf("%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s\n", "Chunk size", "Loop limit", "Bytes/extension", "Duration (ms)", "Filesize (MB)", "HDF Ratio", "Read  speed (MB/s)", "Frequency (kHz)"); 
    if(atoi(argv[2])==0){
        printf("%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s\n", "Chunk size", "Loop limit", "Bytes/extension", "Duration (ms)", "Filesize (MB)", "HDF Ratio", "Read  speed (MB/s)", "Frequency (kHz)");
    };
    loop_read(argv[1], atoi(argv[3]));
    return 0;
}

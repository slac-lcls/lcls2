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

void loop_read(const char* filename){
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    const H5std_string FILE_NAME( filename );
    H5File file(FILE_NAME, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(DATASET_NAME);

    H5T_class_t type_class = dataset.getTypeClass();

    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentDims( dims );


    // Variable length structure to hold data
    hvl_t row_out[1];

    // Define a variable length datatype
    VarLenType int_var_type(&PredType::NATIVE_INT);

    hsize_t row_sel[1] = {1}, offset[2] = {0}, count[2] = {1};
    // int row_out[dims[1]];

    hsize_t chunk_dims[1];
    int rank_chunk;
    DSetCreatPropList cparms = dataset.getCreatePlist();
    rank_chunk = cparms.getChunk(1, chunk_dims);


     for(hsize_t i =0; i<dims[0]; i++){
         DataSpace memspace(1, row_sel);
         dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );
         dataset.read(row_out, int_var_type, memspace, dataspace);
         offset[0]+=1;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    auto fileSize = (float) GetFileSize(FILE_NAME)/1000000; //MB
    float av_speed = 1000*fileSize/(duration); // MB/s
    float av_freq = float(dims[0])/duration;
    float hdf_ratio = 1000000*fileSize/(sizeof(row_out)*dims[0]);

    printf("%-20i%-20i%-20i%-20i%-20.2f%-20.2f%-20.2f%-20.2f\n", chunk_dims[0], dims[0], row_out[0].len*4, duration, fileSize, hdf_ratio, av_speed, av_freq); 


}


int main(int argc,  char *argv[]){
    if(atoi(argv[2])==0){
    printf("%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s\n", "Chunk size", "Loop limit", "Bytes/extension", "Duration (ms)", "Filesize (MB)", "HDF Ratio", "Read  speed (MB/s)", "Frequency (kHz)");
    };
    loop_read(argv[1]);
    return 0;


}

//test code for C++ implemeation of axi stream batcher parser
//https://github.com/slaclab/lcls2-pcie-apps/blob/system_dsp_integration_testing/software/TimeTool/python/TimeToolDev/eventBuilderParser.py

#include <atomic>
#include <string>
#include <iostream>
#include <signal.h>
#include <cstdio>
#include <AxisDriver.h>
#include <stdlib.h>
#include "TimingHeader.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <Python.h>
#include <fstream>
#include <vector>

#define MAX_RET_CNT_C 1000
//static int fd;
std::atomic<bool> terminate;

class eventBuilderParser {
    public:
        std::vector<uint8_t> raw_data;     //pointer to the raw data
        int *main_header;   
        int version;    
        
              
        
        int get_frame_size(){
            return raw_data.size();
        }

        int load_frame(std::vector<uint8_t> &incoming_data){
            raw_data   = incoming_data;
            return 0;        
        }

        int parseArray(){
            version          = raw_data[0] & 15;
            *main_header     = raw_data[0];

            return 0;


        }

        int print_frame(){
            for (int i=0;i<raw_data.size();i=i+1){
                printf("%x ",raw_data[i]);
            }
            printf("\n");
            return 0;
        }



};

int load_file(char *filname, std::vector<uint8_t> &raw_data, int* data_size){

    printf("Starting main \n");

    std::streampos size;
    char * memblock;
   
    std::ifstream file ("/home/sioan/Desktop/timeToolParser/raw_data/p2",std::ios::in|std::ios::binary|std::ios::ate);
    
     if (file.is_open())
      {
        size = file.tellg();
        printf("size = %i \n",size);
        memblock = new char [size];
        file.seekg (0, std::ios::beg);
        file.read (memblock, size);
        file.close();

        std::cout << "the entire file content is in memory \n" ;
        *data_size = size;
        

        
        

        for(int i=0;i<size;i=i+1){
        
        //raw_data[i] = uint8_t(memblock[i]);
            raw_data.push_back(uint8_t(memblock[i]));    
        }
        //raw_data = reinterpret_cast<uint8_t *> (memblock);

        
        delete[] memblock;
        //delete[] raw_data;
      }
      else std::cout << "Unable to open file \n";


    return 0;

}


int main(int argc, char* argv[])
{

      eventBuilderParser my_frame;

      std::vector<uint8_t> raw_data;
      int size;      

      load_file("test",raw_data,&size);


      my_frame.load_frame(raw_data);
    
      my_frame.print_frame();
    

      return 0;

    printf("Done \n");

}

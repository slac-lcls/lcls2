//test code for C++ implemeation of axi stream batcher parser
//https://github.com/slaclab/lcls2-pcie-apps/blob/system_dsp_integration_testing/software/TimeTool/python/TimeToolDev/eventBuilderParser.py
//the above implementation (tentatively) works. this code follows the axi stream event builder protocol located here
//https://confluence.slac.stanford.edu/display/ppareg/AxiStream+Batcher+Protocol+Version+1


#include <atomic>
#include <string>
#include <iostream>
#include <signal.h>
#include <cstdio>
#include <AxisDriver.h>
#include <stdlib.h>
#include "psdaq/service/EbDgram.hh"
#include "AxiBatcherParser.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <Python.h>
#include <fstream>
#include <vector>
#include <typeinfo>

#define MAX_RET_CNT_C 1000

int load_file(std::string test_file, std::vector<uint8_t> &raw_data){

    std::streampos size;
    char * memblock;
   
    std::ifstream file (test_file,std::ios::in|std::ios::binary|std::ios::ate);
    
     if (file.is_open())
      {
        size = file.tellg();
        printf("size = %i \n",(int)size);
        memblock = new char [size];
        file.seekg (0, std::ios::beg);
        file.read (memblock, size);
        file.close();

        std::cout << "the entire file content is in memory \n" ;

        for(int i=0;i<size;i=i+1){
            
            raw_data.push_back(uint8_t(memblock[i]));    
        }

        
        delete[] memblock;
      }
      else std::cout << "Unable to open file \n";    

    return 0;

}


int main(int argc, char* argv[])
{
    
        printf("Starting main \n");

        int c;

        std::string test_file = "/home/sioan/Desktop/timeToolParser/raw_data/p2";
        while((c = getopt(argc, argv, "f:")) != EOF) {
            switch(c) {
                case 'f':
                    test_file = optarg;
                    break;
             }
        }


      eventBuilderParser my_frame;

      std::vector<uint8_t> raw_data;   

      load_file(test_file,raw_data);


      my_frame.load_frame(raw_data);

      my_frame.parse_array();
      
      //my_frame.print_raw();    
      my_frame.print_frame();
    

      return 0;

    printf("Done \n");

}

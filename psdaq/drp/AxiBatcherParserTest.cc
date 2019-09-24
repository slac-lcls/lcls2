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

#define MAX_RET_CNT_C 1000
//static int fd;
std::atomic<bool> terminate;



int main(int argc, char* argv[])
{
    printf("Hello world \n");

    std::streampos size;
    char * memblock;
    uint8_t *raw_data;
   
    std::ifstream file ("/home/sioan/Desktop/timeToolParser/raw_data/p1",std::ios::in|std::ios::binary|std::ios::ate);
    
     if (file.is_open())
      {
        size = file.tellg();
        printf("size = %i \n",size);
        memblock = new char [size];
        file.seekg (0, std::ios::beg);
        printf("size = %i \n",size);
        file.read (memblock, size);
        printf("size = %i \n",size);
        file.close();

        std::cout << "the entire file content is in memory \n" ;
        
        raw_data = reinterpret_cast<uint8_t *>(memblock);

        for (int i=0;i<size;i=i+1){
            printf("%x ",raw_data[i]);
        }
        printf("\n");
        printf("size = %i \n",size);

        delete[] memblock;
        //delete[] raw_data;
      }
      else std::cout << "Unable to open file \n";
      return 0;

    printf("Done \n");

}

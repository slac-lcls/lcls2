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
#include "TimingHeader.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <Python.h>
#include <fstream>
#include <vector>
#include <typeinfo>

#define MAX_RET_CNT_C 1000
//static int fd;
std::atomic<bool> terminate;

class eventBuilderParser {
    public:
        std::vector<uint8_t>                  raw_data;                       //this isn't really the raw data anymore. it's castas a uint8_t now.
        std::vector<uint8_t>                  main_header;   
        std::vector<uint16_t>                 frame_sizes_reverse_order;      //the size of each subframe extracted from the tail
        std::vector<std::vector<uint16_t>>    frame_positions_reverse_order;  //vector of vectors.  N elements long with each element containing a start and an end.
        std::vector<std::vector<uint8_t>>     frame_list;                     //the actual edge positions, camera images, time stamps, etc... will be elements of this array.
                                                                              //I.e. this is the scientific data that gets viewed, analyzed, and published

        int                                   spsft = 16;                     //sptl stand for the size position in the sub frame tail                             
        int                                   version;    



        
        //this method looks at the position of the raw data indicated in the argument, process the data 
        //at that location, and returns the position of the next place to look for the next piece of data
        //it is assumed that the position argument points to the section of the sub-frame tail that contains the subframe length information
        uint16_t frame_to_position(int position){
            
            //uint16_t temp = (raw_data[position+1]<<8) + raw_data[position];
            //printf("position = %d, temp = %d ,raw_data[position] = %d, raw_data[position+1]<<8 = %d\n",position,temp,raw_data[position],raw_data[position+1]<<8);
        
            return (raw_data[position+1]<<8) + raw_data[position]; //+ gets processed before <<
            
        }
              
        
        int get_frame_size(){
            return raw_data.size();
        }

        //this method loads the data into the parser class.  May need to become a copy
        int load_frame(std::vector<uint8_t> &incoming_data){
            raw_data   = incoming_data;
            return 0;        
        }

        //this method does the actual parsing.  Upon completion of this method, the members
        //frame_sizes_reverse_order, frame_position_reverse_order
        int parse_array(){
            version         = raw_data[0] & 15;
            main_header.push_back(raw_data[0]);


            

            //frame_positions_reverse_order.push_back(12);
            frame_sizes_reverse_order.push_back(frame_to_position(raw_data.size()-spsft)); //sptl stand for the size position in the sub frame tail

            std::vector<uint16_t> sub_frame_range = {raw_data.size()-spsft-frame_sizes_reverse_order.back(),raw_data.size()-spsft};
            frame_positions_reverse_order.push_back(sub_frame_range);

            return 0;


        }




        int print_raw(){

            printf("raw data = \n");
            print_vector(raw_data);
            return 0;
        }

        int print_frame(){
    
            printf("sub frame size = \n");
            print_vector(frame_sizes_reverse_order);
            
            printf("sub frame positions = \n");
            print_vector(frame_positions_reverse_order);

            return 0;
        }

        template <class T> int print_vector(std::vector<T> &my_vector){
   
            //printf("%s \n",typeid(r).name());     
             
            for (int i=0;i<my_vector.size();i=i+1){

                printf("%x ",my_vector[i]);
            }
            printf("\n");
            return 0;
        }



};

int load_file(std::string test_file, std::vector<uint8_t> &raw_data){

    std::streampos size;
    char * memblock;
   
    std::ifstream file (test_file,std::ios::in|std::ios::binary|std::ios::ate);
    
     if (file.is_open())
      {
        size = file.tellg();
        printf("size = %i \n",size);
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
    
      my_frame.print_raw();    
      my_frame.print_frame();
    

      return 0;

    printf("Done \n");

}

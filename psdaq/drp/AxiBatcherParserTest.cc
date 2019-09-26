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
        std::vector<std::vector<uint8_t>>     frames;                         //the actual edge positions, camera images, time stamps, etc... will be elements of this array.
                                                                              //I.e. this is the scientific data that gets viewed, analyzed, and published
        std::vector<short>                    is_sub_frame;                   

        int                                   spsft            = 16;          //spsft stand for the size position in the sub frame tail
        int                                   HEADER_WIDTH     = 16;                            
        int                                   version;    
        char*                                 vector_type_name = "St6vector" ;


        
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
        //frame_sizes_reverse_order, frame_position_reverse_order are populated
        int parse_array(){
            version         = raw_data[0] & 15;
            main_header.resize(HEADER_WIDTH);
            std::copy(raw_data.begin(),raw_data.begin()+HEADER_WIDTH,main_header.begin());


            

            //storing frame size
            frame_sizes_reverse_order.push_back(frame_to_position(raw_data.size()-spsft)); //sptl stand for the size position in the sub frame tail

            //storing frame position in raw data
            std::vector<uint16_t> sub_frame_range = {raw_data.size()-spsft-frame_sizes_reverse_order.back(),raw_data.size()-spsft};
            frame_positions_reverse_order.push_back(sub_frame_range);

            
            //storing first parsed frame in frames
            std::vector<uint8_t> frame(sub_frame_range[1]-sub_frame_range[0]);
            std::copy(raw_data.begin()+sub_frame_range[0],raw_data.begin()+sub_frame_range[1],frame.begin());
            frames.push_back(frame);
            

            int parsed_frame_size = frame_positions_reverse_order.size()*HEADER_WIDTH;
            for(std::vector<uint16_t>::iterator it = frame_sizes_reverse_order.begin(); it != frame_sizes_reverse_order.end(); ++it)
                parsed_frame_size += *it;

            

            while(raw_data.size() > (parsed_frame_size+HEADER_WIDTH)){

                //storing frame sizes 
                frame_sizes_reverse_order.push_back(frame_to_position(frame_positions_reverse_order.back()[0]-spsft));


                //storing frame positions in raw data
                std::vector<uint16_t> new_positions;
                new_positions.push_back( frame_positions_reverse_order.back()[0]-spsft-frame_sizes_reverse_order.back());
                new_positions.push_back( frame_positions_reverse_order.back()[0]-spsft);
                frame_positions_reverse_order.push_back(new_positions);


                //parsing frames into new list
                frame.resize(new_positions[1]-new_positions[0]);
                std::copy(raw_data.begin()+new_positions[0],raw_data.begin()+new_positions[1],frame.begin());
                frames.push_back(frame);

                //printf("parsed frame = \n");
                //print_vector(frame);

                


                parsed_frame_size = parsed_frame_size+frame_sizes_reverse_order.back()+HEADER_WIDTH;
                //printf("raw data size = %d, parsed data size = %d \n",raw_data.size(),parsed_frame_size);

            }

            check_for_subframes();

            return 0;


        }


        int check_for_subframes(){

             for (int i = 0 ; i <  frames.size() ; i = i + 1){
                    if(std::equal(frames[i].begin(), frames[i].begin()+2, main_header.begin())){
                     is_sub_frame.push_back(1);                   
                    }
                    else{
                        is_sub_frame.push_back(0);                   
                    }
             }


        return 1;
        }

        int resolve_sub_frames(){
            return 1
        }


        int print_raw(){

            printf("raw data = \n");
            print_vector(raw_data);
            return 0;
        }

        int print_frame(){
   

            printf("\nparsed frames = \n");
            print_vector2d(frames);
            
 
            printf("\nnsub frame size = \n");
            print_vector(frame_sizes_reverse_order);
            
            printf("\nsub frame positions = \n");
            print_vector2d(frame_positions_reverse_order);

            printf("\n vector indicating if it's a sub frame. length = %d \n",is_sub_frame.size());
            print_vector(is_sub_frame);

            
                

            return 0;
        }

        template <class T> int print_vector2d(std::vector<T> &my_vector){
   
            //printf("my_vector data type is %s \n",typeid(my_vector).name());     
            //printf("my_vector[0] data type is %s \n",typeid(my_vector[0]).name());     
             
            for (uint32_t i=0;i<my_vector.size();i=i+1){
                    printf("sub frame %d = [",i);
                    for(uint32_t j=0;j<my_vector[i].size();j=j+1){
                        printf("%d ",my_vector[i][j]);
                    }
                    printf("]\n length = %d \n",my_vector[i].size());
            }
            printf("\n");
            return 0;
        }


        
        template <class T> int print_vector(std::vector<T> &my_vector){
   
            for (uint32_t i=0;i<my_vector.size();i=i+1){
                printf("%d ",my_vector[i]);
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
      
      //my_frame.print_raw();    
      my_frame.print_frame();
    

      return 0;

    printf("Done \n");

}

//test code for C++ implemeation of axi stream batcher parser
//https://github.com/slaclab/lcls2-pcie-apps/blob/system_dsp_integration_testing/software/TimeTool/python/TimeToolDev/eventBuilderParser.py
//the above implementation (tentatively) works. this code follows the axi stream event builder protocol located here
//https://confluence.slac.stanford.edu/display/ppareg/AxiStream+Batcher+Protocol+Version+1

#include "AxiBatcherParser.hh"
#include <atomic>
#include <string>
#include <iostream>
#include <signal.h>
#include <cstdio>
#include <AxisDriver.h>
#include <stdlib.h>
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <Python.h>
#include <fstream>
#include <vector>
#include <typeinfo>
#include <algorithm>

#define MAX_RET_CNT_C 1000

//this method looks at the position of the raw data indicated in the argument, process the data 
//at that location, and returns the position of the next place to look for the next piece of data
//it is assumed that the position argument points to the section of the sub-frame tail that contains the subframe length information
uint16_t eventBuilderParser::frame_to_position(int position){
    
    return (raw_data[position+1]<<8) + raw_data[position]; //+ gets processed before <<
    
}
              
        
int eventBuilderParser::get_frame_size(){
    return raw_data.size();
}

//this method loads the data into the parser class.  May need to become a copy
int eventBuilderParser::load_frame(std::vector<uint8_t> &incoming_data){
    raw_data   = incoming_data;
    return 0;        
}

int eventBuilderParser::clear(){

        raw_data.clear();
        main_header.clear();   
        frame_sizes_reverse_order.clear();
        frame_positions_reverse_order.clear();
        frames.clear();
        sub_frames.clear();
        is_sub_frame.clear();

        return 0;
}

//need to check frames for damage.  this means, among other things, that the size from the sub frame tail isn't the actual size.
bool eventBuilderParser::is_damaged(uint8_t start,uint8_t end){


    if(start>=end || int(end) >=raw_data.size() ){
        clear();
        return true;
    }
    return false;
}

//this method does the actual parsing.  Upon completion of this method, the members
//frame_sizes_reverse_order, frame_position_reverse_order are populated
int eventBuilderParser::parse_array(){

    version         = raw_data[0] & 15;
    main_header.resize(HEADER_WIDTH);
    std::copy(raw_data.begin(),raw_data.begin()+HEADER_WIDTH,main_header.begin());

    //storing frame size
    frame_sizes_reverse_order.push_back(frame_to_position(raw_data.size()-spsft)); //sptl stand for the size position in the sub frame tail

    //storing frame position in raw data
    std::vector<uint16_t> sub_frame_range = {uint16_t(raw_data.size()-spsft-frame_sizes_reverse_order.back()),uint16_t(raw_data.size()-spsft)};
    frame_positions_reverse_order.push_back(sub_frame_range);

    //check for damaged frame
    if(is_damaged(sub_frame_range[0],sub_frame_range[1])){
            clear();
            return 1;//force
        }

        
    
    //storing first parsed frame in frames
    std::vector<uint8_t> frame(sub_frame_range[1]-sub_frame_range[0]);
    std::copy(raw_data.begin()+sub_frame_range[0],raw_data.begin()+sub_frame_range[1],frame.begin());
    frames.push_back(frame);
    

    int parsed_frame_size = frame_positions_reverse_order.size()*HEADER_WIDTH;
    for(std::vector<uint16_t>::iterator it = frame_sizes_reverse_order.begin(); it != frame_sizes_reverse_order.end(); ++it)
        parsed_frame_size += *it;
    

    while(int(raw_data.size()) > (parsed_frame_size+HEADER_WIDTH)){

        //storing frame sizes 
        frame_sizes_reverse_order.push_back(frame_to_position(frame_positions_reverse_order.back()[0]-spsft));


        //storing frame positions in raw data
        std::vector<uint16_t> new_positions = {};
        new_positions.push_back( frame_positions_reverse_order.back()[0]-spsft-frame_sizes_reverse_order.back());
        new_positions.push_back( frame_positions_reverse_order.back()[0]-spsft);
        frame_positions_reverse_order.push_back(new_positions);


        if(is_damaged(new_positions[0],new_positions[1])){
            clear();
            return 1;
        }

        //parsing frames into new list
        frame.resize(new_positions[1]-new_positions[0]);    //source of another core dump
        std::copy(raw_data.begin()+new_positions[0],raw_data.begin()+new_positions[1],frame.begin());
        frames.push_back(frame);



        //keeping track of the index the points to where the raw frame should be broken up.
        parsed_frame_size = parsed_frame_size+frame_sizes_reverse_order.back()+HEADER_WIDTH;

    }

    check_for_subframes();

    return 0;


}


// checks if one of the frame elements is itself an axi batcher sub frame type. 
int eventBuilderParser::check_for_subframes(){

     //before populating for a sub batcher, we need to make sure the frames aren't.  Otherwise there's a segfault in
     //conditional declaration part frames[i].begin+2 that doesn't exist (thanks gdb back trace for pointing here)
     //It's not clear where the damage that requires this check is coming from.  
     //It wasn't present when doing soft triggered testing at 75 KHz, so that indicates it's related to the timing and/or timestamp
     bool damaged_frame = false;

     for (int i = 0 ; i <  int(frames.size()) ; i = i + 1){
        if(frames[i].size()<2){
            damaged_frame = true;
        }

     }

     if(damaged_frame){
        for (int i = 0 ; i <  int(frames.size()) ; i = i + 1){
                is_sub_frame.push_back(0);
        }  

     return 1;
     }
     

     //if the frames aren't damaged, then let's start identifying which frames are axi-batcher frames
     for (int i = 0 ; i <  int(frames.size()) ; i = i + 1){
            if(std::equal(frames[i].begin(), frames[i].begin()+2, main_header.begin())){
            is_sub_frame.push_back(1);

            eventBuilderParser my_sub_frame;

            my_sub_frame.load_frame(frames[i]);
            my_sub_frame.parse_array();

            sub_frames.push_back(my_sub_frame);
           
            }
            else{
                is_sub_frame.push_back(0);                   
            }
     }


return 0;
}

// this will populate the subframe
int eventBuilderParser::resolve_sub_frames(){

    return 1;
}


int eventBuilderParser::print_raw(){

    printf("raw data = \n");
    print_vector(raw_data);
    return 0;
}

int eventBuilderParser::print_frame(){

    //printf("raw parsed frames = \n");
    //print_vector2d(frames);
    
    printf("___________________________\n");
    printf("printing summary data\n");
    printf("nsub frame size = \n");
    print_vector(frame_sizes_reverse_order);
    
    printf("sub frame positions = \n");
    print_vector2d(frame_positions_reverse_order);

    printf("vector indicating if it's a sub frame. length = %d \n",int(is_sub_frame.size()));
    print_vector(is_sub_frame);

    printf("___________________________\n___________________________\n___________________________\n___________________________\n");            
    print_sub_batcher();   

    return 0;
}

template <class T> int eventBuilderParser::print_vector2d(std::vector<T> &my_vector){
     
    for (uint32_t i=0;i<my_vector.size();i=i+1){


            if(is_sub_frame[i]!=1){

                printf("sub frame %d = [",i);
                for(int j=0;j<std::min(int(my_vector[i].size()),32);j=j+1){
                    printf("%d ",my_vector[i][j]);
                }
                printf("]\n length = %d \n",int(my_vector[i].size()));
            }
            else{

                printf("subframe %d is a sub batcher \n",i);

            }
    }
    return 0;
}

int eventBuilderParser::print_sub_batcher(){

    printf("\nPrinting a sub batcher \n");
    for (int i =0; i<int(sub_frames.size());i=i+1){

        sub_frames[i].print_frame();


    }
    
    return 0;
}
        
template <class T> int eventBuilderParser::print_vector(std::vector<T> &my_vector){

    for (int i=0;i<std::min(int(my_vector.size()),32);i=i+1){
        printf("%d ",my_vector[i]);
    }
    printf("\n");
    return 0;
}

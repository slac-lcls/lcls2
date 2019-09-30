//test code for C++ implemeation of axi stream batcher parser
//https://github.com/slaclab/lcls2-pcie-apps/blob/system_dsp_integration_testing/software/TimeTool/python/TimeToolDev/eventBuilderParser.py
//the above implementation (tentatively) works. this code follows the axi stream event builder protocol located here
//https://confluence.slac.stanford.edu/display/ppareg/AxiStream+Batcher+Protocol+Version+1

#pragma once


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


class eventBuilderParser {
    public:
        std::vector<uint8_t>                  raw_data;                       //this isn't really the raw data anymore. it's castas a uint8_t now.
        std::vector<uint8_t>                  main_header;   
        std::vector<uint16_t>                 frame_sizes_reverse_order;      //the size of each subframe extracted from the tail
        std::vector<std::vector<uint16_t>>    frame_positions_reverse_order;  //vector of vectors.  N elements long with each element containing a start and an end.
        std::vector<std::vector<uint8_t>>     frames;                         //the actual edge positions, camera images, time stamps, etc... will be elements of this array.  
                                                                              //I.e. this is the scientific data that gets viewed, analyzed, and published
        std::vector<eventBuilderParser>       sub_frames; 
        
        std::vector<short>                    is_sub_frame;                   

        int                                   spsft               = 16;       //spsft stand for the size position in the sub frame tail
        int                                   HEADER_WIDTH        = 16;                            
        int                                   version;    
        char*                                 vector_type_name    = "St6vector" ;

        bool                                  DEBUG;             


        
        //this method looks at the position of the raw data indicated in the argument, process the data 
        //at that location, and returns the position of the next place to look for the next piece of data
        //it is assumed that the position argument points to the section of the sub-frame tail that contains the subframe length information
        uint16_t frame_to_position(int position);
        int get_frame_size();

        //this method loads the data into the parser class.  May need to become a copy
        int load_frame(std::vector<uint8_t> &incoming_data);

        //this method does the actual parsing.  Upon completion of this method, the members
        //frame_sizes_reverse_order, frame_position_reverse_order are populated
        int parse_array();        // checks if one of the frame elements is itself an axi batcher sub frame type. 
        int check_for_subframes();

        // this will populate the subframe
        int resolve_sub_frames();
        int print_raw();
        int print_frame();

        template <class T> int print_vector2d(std::vector<T> &my_vector);

        int print_sub_batcher();
        
        template <class T> int print_vector(std::vector<T> &my_vector);


};

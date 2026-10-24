/**
 * @file
 * @brief eventBuilderParser, a parser for AxiStream batcher frames (a C++ port of a Python parser, per the file comment).
 */
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
#include "psdaq/aes-stream-drivers/AxisDriver.h"
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

/** Defined as 1000; not used in AxiBatcherParser.cc. */
#define MAX_RET_CNT_C 1000


/** Parser for AxiStream batcher frames: splits a byte buffer into subframes by walking the subframe tails back from the end, and recursively parses subframes that are themselves batcher frames. */
class eventBuilderParser {
    public:
        /** Bytes of the frame being parsed, set by load_frame(). */
        std::vector<uint8_t>                  raw_data;                       //this isn't really the raw data anymore. it's castas a uint8_t now.
        std::vector<uint8_t>                  main_header;  ///< Copy of the first HEADER_WIDTH bytes of raw_data, set by parse_array().
        /** Subframe sizes read from the subframe tails, last subframe first. */
        std::vector<uint16_t>                 frame_sizes_reverse_order;      //the size of each subframe extracted from the tail
        /** Start and end byte offsets of each subframe in raw_data, last subframe first. */
        std::vector<std::vector<uint16_t>>    frame_positions_reverse_order;  //vector of vectors.  N elements long with each element containing a start and an end.
        /** Bytes of each subframe, last subframe first. */
        std::vector<std::vector<uint8_t>>     frames;                         //the actual edge positions, camera images, time stamps, etc... will be elements of this array.
                                                                              //I.e. this is the scientific data that gets viewed, analyzed, and published
        std::vector<eventBuilderParser>       sub_frames;  ///< Parsers for the subframes that are themselves batcher frames, filled by check_for_subframes().

        std::vector<short>                    is_sub_frame;  ///< One entry per subframe: 1 if it starts with the same two bytes as main_header (parsed as a nested batcher frame), else 0.

        /** Offset (16) from the end of a subframe to its 16-bit size field; the code comment calls it the size position in the subframe tail. */
        int                                   spsft               = 16;       //spsft stand for the size position in the sub frame tail
        int                                   HEADER_WIDTH        = 16;  ///< Width in bytes (16) of the main header copied by parse_array() and of each subframe tail counted while parsing.
        int                                   version;  ///< Low 4 bits of the first byte of raw_data, set by parse_array().

        bool                                  DEBUG;  ///< Debug flag; not initialized and not read in AxiBatcherParser.cc.



        //this method looks at the position of the raw data indicated in the argument, process the data
        //at that location, and returns the position of the next place to look for the next piece of data
        //it is assumed that the position argument points to the section of the sub-frame tail that contains the subframe length information
        /** Return the little-endian 16-bit value at byte position of raw_data (a subframe size field). */
        uint16_t frame_to_position(int position);
        /** Return the size of raw_data in bytes. */
        int get_frame_size();

        //this method loads the data into the parser class.  May need to become a copy
        /** Copy incoming_data into raw_data; returns 0. */
        int load_frame(std::vector<uint8_t> &incoming_data);

        //this method does the actual parsing.  Upon completion of this method, the members
        //frame_sizes_reverse_order, frame_position_reverse_order are populated
        /** Parse raw_data: record the version and main header, walk back from the end reading the size, offsets and bytes of each subframe, then call check_for_subframes(). Returns 0, or 1 after clear() if a frame looks damaged. */
        int parse_array();        // checks if one of the frame elements is itself an axi batcher sub frame type.
        /** Flag each frame as a nested batcher frame or not (by comparing its first two bytes with main_header) and parse the nested ones into sub_frames. Returns 1, flagging none, if any frame is shorter than 2 bytes; otherwise 0. */
        int check_for_subframes();

        // this will populate the subframe
        /** Does nothing; returns 1. */
        int resolve_sub_frames();

        //pgpread_timetool sub frames keep growing.  this should prevent that.
        /** Empty all member vectors; returns 0. */
        int clear();

        //checking for damage in frame
        /** Return true, after calling clear(), if start is not below end or end is not inside raw_data; otherwise false. start and end are uint8_t, so the 16-bit offsets passed by parse_array() are truncated. */
        bool is_damaged(uint8_t start,uint8_t end);

        //printing for debugging
        /** Print up to the first 32 bytes of raw_data to stdout; returns 0. */
        int print_raw();
        /** Print the subframe sizes, positions and nested-batcher flags, then the nested batchers, to stdout; returns 0. */
        int print_frame();

        /** Print up to 32 elements of each entry of my_vector (or a note for entries flagged as nested batchers) to stdout; returns 0. */
        template <class T> int print_vector2d(std::vector<T> &my_vector);

        /** Call print_frame() on each nested batcher parser; returns 0. */
        int print_sub_batcher();

        /** Print up to the first 32 elements of my_vector to stdout; returns 0. */
        template <class T> int print_vector(std::vector<T> &my_vector);


};

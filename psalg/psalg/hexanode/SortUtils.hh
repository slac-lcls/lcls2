#ifndef SORTUTILS_H
#define SORTUTILS_H

//-----------------------------

#include <stdio.h>     // FILE

//#include "psalg/hexanode/resort64c.hh"
#include "roentdek/resort64c.h"

//-----------------------------

__int32 my_kbhit(void);

void readline_from_config_file(FILE * ffile, char * text, __int32 max_len);

int read_int(FILE * ffile);

double read_double(FILE * ffile);

bool read_config_file(const char * name, sort_class *& sorter, int& command,
                      double& offset_sum_u, double& offset_sum_v, double& offset_sum_w,
                      double& w_offset, double& pos_offset_x, double& pos_offset_y);

bool read_calibration_tables(const char * filename, sort_class * sorter);

bool create_calibration_tables(const char* filename, sort_class* sorter);

bool sorter_scalefactors_calibration_map_is_full_enough(sort_class* sorter);

//-----------------------------

#endif

//-----------------------------

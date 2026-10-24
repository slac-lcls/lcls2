/**
 * @file
 * @brief Declares helpers for the RoentDek sort_class: reading sorter config and calibration tables, writing calibration tables, and keyboard polling.
 */
#ifndef SORTUTILS_H
#define SORTUTILS_H

//-----------------------------

#include <stdio.h>     // FILE

//#include "psalg/hexanode/resort64c.hh"
#include "roentdek/resort64c.h"

//-----------------------------

/** Read one character from stdin with a 0.1 s timeout and return it, or 0 if none was read. It temporarily changes the terminal settings of fd 0; the code clears all local-mode flags (it uses !ICANON, which is 0) and restores them afterwards. */
__int32 my_kbhit(void);

/**
 * Read the next token from ffile into text (at most max_len-1 characters): skip leading spaces, tabs, newlines and CR characters, skip comments that start with '/' and run to the end of the line, and stop at a space, tab, newline or comma.
 * End of file is not detected.
 */
void readline_from_config_file(FILE * ffile, char * text, __int32 max_len);

/** Read the next token with readline_from_config_file() and return atoi() of it. */
int read_int(FILE * ffile);

/** Read the next token with readline_from_config_file() and return atof() of it. */
double read_double(FILE * ffile);

/**
 * Read a sorter configuration from the text file name, in a fixed order, into sorter and the output arguments: command, HEX and common-start flags, channel numbers (stored minus 1), offsets, time-sum half widths, scale factors (stored halved), w_offset, runtimes, MCP radius, dead times, correction flags, and a check value that must be 88888.
 * Returns false if the file cannot be opened, if command is -1 (reading stops there), or if the check value is wrong (then sorter is deleted and set to 0); otherwise true. Progress is printed to stdout.
 */
bool read_config_file(const char * name, sort_class *& sorter, int& command,
                      double& offset_sum_u, double& offset_sum_v, double& offset_sum_w,
                      double& w_offset, double& pos_offset_x, double& pos_offset_y);

/**
 * Read sum- and position-correction points (a count, then x y pairs, for layers U, V and, with use_HEX, W) from filename and add them to the sorter's signal correctors when use_sum_correction or use_pos_correction is set.
 * Returns false if filename or sorter is null or the file cannot be opened, otherwise true.
 */
bool read_calibration_tables(const char * filename, sort_class * sorter);

/**
 * Run sorter->do_calibration() and write the sum-walk correction points of layers U, V (and W with use_HEX) and the position-walk points of U, V and W to filename as text (a count line, then x y pairs); the position-walk counts are 0 unless use_HEX is set.
 * Returns false if sorter or filename is null, otherwise true; a failed fopen() is not checked.
 */
bool create_calibration_tables(const char* filename, sort_class* sorter);

/** Return sorter->scalefactors_calibrator->map_is_full_enough(), or false if sorter is null. The check itself is in the RoentDek library; not visible here. */
bool sorter_scalefactors_calibration_map_is_full_enough(sort_class* sorter);

//-----------------------------

#endif

//-----------------------------

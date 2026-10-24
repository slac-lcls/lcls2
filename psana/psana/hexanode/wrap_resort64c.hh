/**
 * @file
 * @brief Declares test_resort(), a small test of the RoentDek sort_class included through resort64c.hh.
 */
#ifndef WRAP_RESORT64C_H
#define WRAP_RESORT64C_H

//#include "roentdek/resort64c.h"
// or the same through the wrapper
#include "resort64c.hh"

/** Print messages to stdout, create a sort_class, set a few fields (common start mode, HEX and MCP use, MCP radius 7, time-sum half widths 10) and delete it again. */
void test_resort();

#endif

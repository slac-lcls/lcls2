/**
 * \file xtc_io_api_cpp.hh
 * @brief Internal utility functions in C++ for XTC iterator.
 *  Created on: Feb 16, 2020
 *      Author: tonglin
 */

#ifndef XTC_IO_API_CPP_HH_
#define XTC_IO_API_CPP_HH_

//#include <stdio.h>
//#include <stdlib.h>
#include <string>
#include <vector>
using namespace std;

vector<string> str_tok(const char* str, const char* delimiters_str);
string tok_to_str(vector<string> token_list);
void print_path(vector<string>vec);
void cc_extern_test_root(void* root_obj);
//xtc_object* xtc_obj_new(int fd, //const
//      XtcFileIterator* fileIter, //const
//      DebugIter* dbgiter, //const
//      Dgram* dg, ////dg = fileIter.next()
//      const char* obj_path_abs);
// process(Xtc* xtc): iterate(dg->xtc)

#endif /* XTC_IO_API_CPP_HH_ */

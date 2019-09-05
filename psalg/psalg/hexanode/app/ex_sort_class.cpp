//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: ex_sort_class.cpp 13182 2017-02-22 20:25:58Z davidsch@SLAC.STANFORD.EDU $
//
// Description:
//	Test class 
//
// Author:
//      Mikhail Dubrovin
//------------------------------------------------------------------------

// ~/lib/hexanode-lib/sort_non-LMF_from_1_detector/resort64c.h
// /reg/common/package/hexanodelib/0.0.1/x86_64-centos7-gcc485/resort64c.h

#include "hexanode_proxy/resort64c.h"

#include <string>
#include <iostream>

using namespace std;
//------------------

int test_sort_class()
{
  cout << "test_sort_class\n";
  sort_class* sorter = new sort_class();

  sorter -> common_start_mode = true;
  sorter -> use_HEX = true;
  sorter -> use_MCP = true;
  sorter -> MCP_radius = 7; // in mm +3mm larger
  sorter -> uncorrected_time_sum_half_width_u = 10;
  sorter -> uncorrected_time_sum_half_width_v = 10;
  sorter -> uncorrected_time_sum_half_width_w = 10;

  delete sorter;

  return 0;
}

//------------------

int main (int argc, char* argv[])
{  
  cout << "Number of input arguments = " << argc << endl; 
  // atoi(argv[1])==1) 
  
  test_sort_class();

  return 0;
}

//------------------

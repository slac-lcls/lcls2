

// #include "hexanode/resort64c.h"
#include "psalg/hexanode/wrap_resort64c.hh"

#include <iostream>
using namespace std;

void test_resort()
{
  cout << "In src/wrap_resort64c.cc which includes resort64c.h through the /wrap_resort64c.hh\n"; 

  cout << "test sort_class\n";
  sort_class* sorter = new sort_class();

  sorter -> common_start_mode = true;
  sorter -> use_HEX = true;
  sorter -> use_MCP = true;
  sorter -> MCP_radius = 7; // in mm +3mm larger
  sorter -> uncorrected_time_sum_half_width_u = 10;
  sorter -> uncorrected_time_sum_half_width_v = 10;
  sorter -> uncorrected_time_sum_half_width_w = 10;

  delete sorter;
  cout << "Done\n";
}

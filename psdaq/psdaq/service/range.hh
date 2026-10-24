/**
 * @file
 * @brief getRange(), which expands a list of numbers and number ranges into integers.
 */
#pragma once

#include <string>
#include <vector>

namespace Pds
{
  /** Append to data every integer in input, where each item is a number or an inclusive range written N-M (items are found by regular expression search, so any separator works). */
  void getRange(const std::string& input, std::vector<int>& data);
}

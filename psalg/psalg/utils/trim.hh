/**
 * @file
 * @brief Declares psalg string helpers that remove whitespace: ltrim, rtrim, trim and strip.
 */
#pragma once

#include <string>

namespace psalg
{

/** Return a copy of str without leading whitespace (space, newline, carriage return, tab, form feed, vertical tab); empty if str is all whitespace. */
std::string ltrim(const std::string& str); // Remove leading whitepace
/** Return a copy of str without trailing whitespace (space, newline, carriage return, tab, form feed, vertical tab); empty if str is all whitespace. */
std::string rtrim(const std::string& str); // Remove trailing whitepace
/** Return rtrim(ltrim(str)). */
std::string trim (const std::string& str); // Remove leading and trailing whitepace
/** Return a copy of str with every whitespace character removed (std::isspace in the classic locale). */
std::string strip(const std::string& str); // Remove all whitepace

};

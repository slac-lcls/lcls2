/**
 * @file
 * @brief get_kwargs(), which parses a comma-separated key=value string into a map.
 */
#pragma once

#include <string>
#include <map>

/** Split kwargs_str at commas and store each key=value pair (key and value trimmed of surrounding white space) in kwargs, overwriting existing keys. Logs and throws a std::string if an item has no equal sign. */
void get_kwargs(const std::string& kwargs_str, std::map<std::string,std::string>& kwargs);

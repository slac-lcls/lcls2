#pragma once

#include <string>
#include <map>

void get_kwargs(const std::string& kwargs_str, std::map<std::string,std::string>& kwargs);

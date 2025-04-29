#pragma once

#include <string>

namespace psalg
{

std::string ltrim(const std::string& str); // Remove leading whitepace
std::string rtrim(const std::string& str); // Remove trailing whitepace
std::string trim (const std::string& str); // Remove leading and trailing whitepace
std::string strip(const std::string& str); // Remove all whitepace

};

// From https://www.techiedelight.com/trim-string-cpp-remove-leading-trailing-spaces/

#include "psalg/utils/trim.hh"
#include <iostream>
#include <string>
#include <algorithm>

namespace psalg
{

const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string& str)
{
    size_t first = str.find_first_not_of(WHITESPACE);
    return (first == std::string::npos) ? "" : str.substr(first);
}

std::string rtrim(const std::string& str)
{
    size_t last = str.find_last_not_of(WHITESPACE);
    return (last == std::string::npos) ? "" : str.substr(0, last + 1);
}

std::string trim(const std::string& str)
{
    return rtrim(ltrim(str));
}

};

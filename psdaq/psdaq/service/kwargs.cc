#include "kwargs.hh"

#include "psalg/utils/SysLog.hh"

#include <string>
#include <sstream>

using logging = psalg::SysLog;

static
std::string trim(const std::string& str)
{
    const auto first = str.find_first_not_of(" ");
    if (first == std::string::npos)  return "";
    const auto last = str.find_last_not_of(" ");
    return str.substr(first, last - first + 1);
}

void get_kwargs(const std::string& kwargs_str, std::map<std::string,std::string>& kwargs) {
    std::istringstream ss(kwargs_str);
    std::string kwarg;
    while (getline(ss, kwarg, ',')) {
        auto pos = kwarg.find("=", 0);
        if (pos == std::string::npos) {
            logging::critical("Keyword argument with no equal sign");
            throw "Keyword argument with no equal sign: "+kwargs_str;
        }
        std::string key = trim(kwarg.substr(0,pos));
        std::string value = trim(kwarg.substr(pos+1,kwarg.length()));
        //std::cout << "kwarg = '" << kwarg << "' key = '" << key << "' value = '" << value << "'" << std::endl;
        kwargs[key] = value;
    }
}

#include "kwargs.hh"

#include "psalg/utils/SysLog.hh"
#include "psalg/utils/trim.hh"

#include <string>
#include <sstream>
//#include <iostream>

using logging = psalg::SysLog;

void get_kwargs(const std::string& kwargs_str, std::map<std::string,std::string>& kwargs) {
    std::istringstream ss(kwargs_str);
    std::string kwarg;
    while (getline(ss, kwarg, ',')) {
        auto pos = kwarg.find("=", 0);
        if (pos == std::string::npos) {
            logging::critical("Keyword argument with no equal sign");
            throw "Keyword argument with no equal sign: "+kwargs_str;
        }
        std::string key = psalg::trim(kwarg.substr(0,pos));
        std::string value = psalg::trim(kwarg.substr(pos+1,kwarg.length()));
        //std::cout << "kwarg = '" << kwarg << "' key = '" << key << "' value = '" << value << "'" << std::endl;
        kwargs[key] = value;
    }
}

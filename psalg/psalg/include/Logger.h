#ifndef PSALG_LOGGER_H
#define PSALG_LOGGER_H

//---------------------------------------------------
// Created 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------

#include <string>
#include <iostream> // cout, puts etc.
#include <sstream>   // stringstream, streambuf
//#include <fstream>

namespace psalg {

//-------------------

  enum        LEVELS    {INFO=0, WARNING,   ERROR,   DEBUG};
  std::string STLEVS[]={"INFO", "WARNING", "ERROR", "DEBUG"};

  void MsgLog(const std::string& name, const unsigned& level, const std::string& msg) {
    std::cout << name << " " << STLEVS[level] << " " << msg <<'\n';
  }

  void MsgLog(const std::string& name, const unsigned& level, std::ostream& ss) {
    //std::ostringstream oss; oss<<ss.rdbuf();
    std::string s = static_cast<std::ostringstream&>(ss).str();
    std::cout << name << " " << STLEVS[level] << " " << s <<'\n';
  }

  /*
  void MsgLog(const std::string& name, const unsigned& level, std::stringstream& ss) {
    std::cout << name << " " << STLEVS[level] << " " << ss.str() <<'\n';
  }
  */

//-------------------

} // namespace psalg

#endif // PSALG_LOGGER_H

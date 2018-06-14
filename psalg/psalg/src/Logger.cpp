
//-------------------
#include "psalg/include/Logger.h" // MsgLog, Logger, LOGPRINT, LOGMSG
//-------------------
 
//#include <iostream> // cout
#include <stdexcept>

//namespace psalg {

/*
std::string STLEVS(LEVELS level) {
      std::string stlevs[]={"INFO", "WARNING", "ERROR", "DEBUG", LAST_LEVEL = nolog};
      return stlevs[level];
}
*/
//-------------------

Logger* Logger::_pinstance = NULL; // init static pointer for singleton
//-------------------

std::string Logger::level_to_name(const Logger::LEVEL& level)
{
  switch (level) {
    case DEBUG:   return "DEBUG";
    case TRACE:   return "TRACE";
    case INFO:    return "INFO";
    case WARNING: return "WARNING";
    case ERROR:   return "ERROR";
    case FATAL:   return "FATAL";
    case NOLOG:
    default:      return "NOLOG";
  }
}

//-------------------

Logger::LEVEL Logger::name_to_level(const std::string& levname) {
  if      (levname == "TRACE")   return TRACE;
  else if (levname == "DEBUG" )  return DEBUG;
  else if (levname == "WARNING") return WARNING;
  else if (levname == "INFO")    return INFO;
  else if (levname == "ERROR")   return ERROR;
  else if (levname == "FATAL")   return FATAL;
  else if (levname == "NOLOG")   return NOLOG;
  else throw std::out_of_range("unexpected logging level name");
}

//Logger* Logger::instance(){
//    if(!_pinstance) _pinstance = new Logger();
//    return _pinstance;
//}

//-------------------

//} // namespace psalg

//-------------------

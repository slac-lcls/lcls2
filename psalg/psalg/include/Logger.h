#ifndef PSALG_LOGGER_H
#define PSALG_LOGGER_H

//---------------------------------------------------
// Created 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------

#include <string>
#include <iostream> // cout, puts etc.
#include <sstream>   // stringstream, streambuf
//#include <fstream>


enum LEVELS {INFO=0, WARNING, ERROR, DEBUG, NOLOG, LAST_LEVEL=NOLOG};


//namespace psalg {

//-------------------

  //extern char *stlevels[];

  //static std::string STLEVS[4]={"INFO", "WARNING", "ERROR", "DEBUG"};
  //std::string STLEVS(LEVELS);

/*
  inline std::string STLEVS(LEVELS level) {
      std::string stlevs[]={"INFO", "WARNING", "ERROR", "DEBUG"};
      return stlevs[level];
  }
*/

  inline char* STLEVS(LEVELS level) {
      char* stlevs[]={(char*)"INFO", (char*)"WARNING", (char*)"ERROR", (char*)"DEBUG"};
      return stlevs[level];
  }

  inline void MsgLog(const std::string& name, const LEVELS level, const std::string& msg) {
    std::cout << name << " " << STLEVS(level) << " " << msg <<'\n';
  }

  inline void MsgLog(const std::string& name, const LEVELS level, std::ostream& ss) {
    std::string s = static_cast<std::ostringstream&>(ss).str();
    std::cout << name << " " << STLEVS(level) << " " << s <<'\n'; // << " __LINE__:" << __LINE__ << " __FILE__:" << __FILE__ <<'\n';
  }

  //inline void MsgLog(const std::string& name, const LEVELS level, std::stringstream& ss) {
  //  //std::stringstream& s = static_cast<std::ostringstream&>(ss).str();
  //  std::cout << name << " " << STLEVS(level) << " " << ss.str() << '\n';
  //}

#ifdef MSGLOG
#undef MSGLOG
#endif

#define MSGLOG(logger,sev,msg) \
  std::stringstream ss; ss<<msg; \
  MsgLog(logger,sev,ss);

//-------------------

//-------------------
// #include "psalg/include/Logger.h" // MsgLog, Logger, LOGPRINT, LOGMSG
//-------------------
 
//#include <iostream> // cout

class Logger {

enum LEVEL {INFO=0, WARNING, ERROR, FATAL, TRACE, DEBUG, NOLOG, LAST_LEVEL=NOLOG};


public:
  static Logger* instance() {
    if(!_pinstance) _pinstance = new Logger();
    return _pinstance;
  }

  inline void print(){_counter++; std::cout << "Logger::print() " << _counter << '\n';};
  inline void logmsg(const std::string& s){_counter++; std::cout << "msg:" << _counter << ' ' << s << '\n';}; 
  inline void set_logname(const std::string& logname) {_logname=logname;}
  inline void set_level(const LEVEL& level) {_level=level;}
  std::string level_to_name(const LEVEL& level);
  LEVEL name_to_level(const std::string& name);

private:
  // !!! Private - it can not be called from outside
  Logger() : _counter(0), _logname(""), _level(DEBUG) {std::cout << "Single instance of class Logger\n";}; 
 
  virtual ~Logger(){}; 

  static Logger* _pinstance; // !!! Singleton instance

  unsigned    _counter; // record counter
  std::string _logname;    // logger name
  LEVEL       _level;   // level of messages

  // Copy constructor and assignment are disabled by default
  Logger(const Logger&);
  Logger& operator = (const Logger&);
};

//-------------------

// Shortcuts:
#define LOGGER  (*Logger::instance())
#define LOGPRINT  Logger::instance()->print
#define LOGMSG(S) Logger::instance()->logmsg(S)

//-------------------
//} // namespace psalg

#endif // PSALG_LOGGER_H

//-------------------

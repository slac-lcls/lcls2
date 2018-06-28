#ifndef PSALG_LOGGER_H
#define PSALG_LOGGER_H

//---------------------------------------------------
// Created 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------
//-------------------
/**  Usage mainstream:
 *   =================
 *   #include "psalg/include/Logger.hh" // MSG, MSGLOG, LOGGER, MSGSTREAM
 *   
 *   MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
 *   LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format
 *
 *   MSG(DEBUG,   "Test MSG DEBUG" << " as stream");
 *   MSG(TRACE,   "Test MSG TRACE");
 *   MSG(INFO,    "Test MSG INFO");
 *   MSG(WARNING, "Test MSG WARNING");
 *   MSG(ERROR,   "Test MSG ERROR");
 *   MSG(FATAL,   "Test MSG FATAL");
 *   MSG(NOLOG,   "Test MSG NOLOG");
 *   MSGSTREAM(INFO, out){out << "Test MSGSTREAM INFO"; out << " which supports block of output statements as a single message";} 
 *   
 *   Usage optional
 *   ==============
 *   LOGGER.setLevel(LL::DEBUG);
 *   LOGGER.setTimeFormat("%Y-%m-%d %H:%M:%S.%f");
 *   LOGGER.loggerInfo(std::cout); // or Logger::Logger::instance()->loggerIinfo(out);
 */

//-------------------
 
//extern "C" {}

#include <vector>
#include <string>
#include <sstream>   // stringstream, streambuf
//#include <fstream>

//-------------------

/*
namespace {
  void formattedTime(std::string fmt, std::ostream& out);
  const char* tstampNow(std::string fmt="%Y-%m-%d %H:%M:%S.f");
}
*/

//-------------------

namespace Logger {

//-------------------

class LogStream;
class LogRecord;
class LogHandler;
class LogHandlerStdStreams;

class Logger {

public:

  enum LEVEL {DEBUG=0, TRACE, INFO, WARNING, ERROR, FATAL, NOLOG, LAST_LEVEL=NOLOG, NUM_LEVELS=LAST_LEVEL+1};

  char* LEVELCN[NUM_LEVELS];
  char* LEVELC3[NUM_LEVELS];
  char  LEVELC1[NUM_LEVELS];

  static Logger* instance() {
    if(!_pinstance) _pinstance = new Logger();
    return _pinstance;
  }

  void logmsg(const LogStream& ss, const LEVEL& sev=DEBUG);

  void loggerInfo(std::ostream& out);
  const char* tstampStart() {return _tstamp_start.c_str();}

  inline void setLogname(const std::string& logname) {_logname=logname;}
  inline void setLevel(const LEVEL& level) {_level=level;}
  inline bool logging(const LEVEL& sev) {return (sev >= _level);}
  const char* levelToName(const LEVEL& level);
  LEVEL name_to_level(const std::string& name);
  void log(const LogRecord& rec);
  const unsigned counter() {return _counter;} 
  void setLogger(const LEVEL& level=DEBUG, const std::string& timefmt="%H:%M:%S.%f");

  /// add a handler for the messages, takes ownership of the object
  void addHandler(LogHandler* handler) {_handlers.push_back(handler);}
  void setTimeFormat(const std::string& timefmt="%Y-%m-%d %H:%M:%S.%f");

private:
  typedef std::vector<LogHandler*> HandlerList;

  Logger();
 
  virtual ~Logger(); 
  void _initLevelNames();

  static Logger* _pinstance; // !!! Singleton instance

  unsigned    _counter; // record counter
  std::string _logname; // logger name
  LEVEL       _level;   // level of messages
  //const char* _tstamp_start; // start logeer timestamp
  const std::string _tstamp_start; // start logeer timestamp

  HandlerList _handlers;

  Logger(const Logger&);
  Logger& operator = (const Logger&);
};

//-------------------
//-------------------
//-------------------

class LogStream : public std::stringstream {
public:

  typedef Logger::LEVEL level_t;

  LogStream(const std::string& logname, const level_t& sev, const char* file=0, int line=-1);
  virtual ~LogStream(){_emit_content();}
  std::ostream& logger_ostream() {return *this;}
  inline const char* file() const {return _filename;}
  inline const int line() const {return _linenum;}

  /// get the state of the stream
  bool ok() const {return _ok;}

  // set the state of the stream to "not OK"
  void finish(){_ok=false;}

private:
  std::string _logname;
  level_t _sev;
  const char* _filename;
  int _linenum;
  bool _ok;
  void _emit_content() const;

  LogStream(const LogStream&);             // Copy Constructor
  LogStream& operator= (const LogStream&); // Assignment op
};

//-------------------
//-------------------
//-------------------

class LogRecord {

public:

  typedef Logger::LEVEL level_t;

  LogRecord(const std::string& logger,
            const level_t& level,
            const char* filename,
            int linenum,
            std::streambuf* msgbuf)
    : _logger(logger), _level(level), _filename(filename), _linenum(linenum), _msgbuf(msgbuf) {}

  ~LogRecord() {}

  /// get logger name
  const std::string& logger() const {return _logger;}

  /// get message log level
  const level_t level() const {return _level;}

  /// get message location
  const char* file() const {return _filename;}
  int line() const {return _linenum;}

  /// get the stream for the specified log level
  std::streambuf* msgbuf() const {return _msgbuf;}

private:

  const std::string& _logger;
  const level_t _level;
  const char* _filename;
  const int _linenum;
  std::streambuf* _msgbuf;

  LogRecord(const LogRecord&);
  LogRecord& operator= (const LogRecord&);
};

//-------------------
//-------------------
//-------------------

class LogFormatter {

public:

  typedef Logger::LEVEL level_t;

  LogFormatter(const std::string& fmt="", const std::string& timefmt=""); //%Y-%m-%d %H:%M:%S.%f");

  virtual ~LogFormatter() {}

  /// add format
  virtual void addFormat(const level_t& level, const std::string& fmt);

  virtual void setTimeFormat(const std::string& timefmt=""); // "%Y-%m-%d %H:%M:%S.%f";

  /// format message to the output stream
  virtual void format(const LogRecord& rec, std::ostream& out);

protected:

  /// get a format string for a given level
  virtual const std::string& getFormat(const level_t& level) const;

private:

  std::string _timefmt;
  std::string _fmtMap[level_t::NUM_LEVELS];

  LogFormatter(const LogFormatter&);
  LogFormatter& operator= (const LogFormatter&);
};

//-------------------
//-------------------
//-------------------

class LogHandler {

public:

  virtual ~LogHandler();

  /// attaches the formatter, will be owned by handler
  virtual void setFormatter(LogFormatter* formatter);

  /// get the stream for the specified log level
  virtual bool log(const LogRecord& record) const=0;

  /// get formatter
  LogFormatter& formatter() const;

protected:

  LogHandler();

private:

  mutable LogFormatter* _formatter;

  LogHandler(const LogHandler&);
  LogHandler& operator= (const LogHandler&);
};

//-------------------
//-------------------
//-------------------

class LogHandlerStdStreams : public LogHandler {

public:
  LogHandlerStdStreams();
  virtual ~LogHandlerStdStreams();

  /// get the stream for the specified log level
  virtual bool log(const LogRecord& record) const;
};

//-------------------
//-------------------
//-------------------

} // namespace Logger

//-------------------
//-------------------
//-------------------

// Shortcuts:
#define LL Logger::Logger
#define LOGGER (*LL::instance())
#define __MACROPARS __FILE__,__LINE__

#ifdef MSGLOG
#undef MSGLOG
#endif
#define MSGLOG(logger,sev,msg) \
  if (LOGGER.logging(LL::sev)){ \
    Logger::LogStream _log_stream(logger,LL::sev,__MACROPARS); _log_stream.logger_ostream() << msg; \
  }

#ifdef MSG
#undef MSG
#endif
#define MSG(sev,msg) \
  if (LOGGER.logging(LL::sev)){ \
    Logger::LogStream _log_stream(std::string(),LL::sev,__MACROPARS); _log_stream << msg; \
  }

#ifdef MSGSTREAM
#undef MSGSTREAM
#endif
#define MSGSTREAM(sev,strm) \
    for(Logger::LogStream strm(std::string(),LL::sev,__MACROPARS); strm.ok(); strm.finish())

//-------------------

#endif // PSALG_LOGGER_H

//-------------------

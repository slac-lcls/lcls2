#ifndef PSALG_LOGGER_H
#define PSALG_LOGGER_H

//---------------------------------------------------
// Created 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------
//-------------------
// #include "psalg/include/Logger.h" // MSG, MSGLOG, LOGGER, Logger
//-------------------
 
//extern "C" {}

#include <vector>
#include <string>
#include <iostream> // cout, puts etc.
#include <sstream>   // stringstream, streambuf
//#include <fstream>

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

  void logger_info(std::ostream& out);

  inline void set_logname(const std::string& logname) {_logname=logname;}
  inline void set_level(const LEVEL& level) {_level=level;}
  inline bool logging(const LEVEL& sev) {return (sev >= _level);}
  const char* level_to_name(const LEVEL& level);
  LEVEL name_to_level(const std::string& name);
  void log(const LogRecord& rec);
  const unsigned counter() {return _counter;} 

  /// add a handler for the messages, takes ownership of the object
  void addHandler(LogHandler* handler) {_handlers.push_back(handler);}

private:
  typedef std::vector<LogHandler*> HandlerList;

  Logger();
 
  virtual ~Logger(); 
  void _init_level_names();

  static Logger* _pinstance; // !!! Singleton instance

  unsigned    _counter; // record counter
  std::string _logname; // logger name
  LEVEL       _level;   // level of messages

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

private:
  std::string _logname;
  level_t _sev;
  const char* _filename;
  int _linenum;
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

  LogFormatter(const std::string& fmt="", const std::string& timefmt="");

  virtual ~LogFormatter() {}

  /// add format
  virtual void addFormat(const level_t& level, const std::string& fmt);

  /// format message to the output stream
  virtual void format(const LogRecord& rec, std::ostream& out);

protected:

  /// get a format string for a given level
  virtual const std::string& getFormat(const level_t& level) const;

private:

  std::string _timefmt ;
  std::string _fmtMap[level_t::NUM_LEVELS] ;

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

protected:

  LogHandler();
  /// get formatter
  LogFormatter& formatter() const;

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

//    LOGGER.logmsg(_log_stream, Logger::Logger::sev);	
//_log_stream.logger_ostream() << msg; 
//    std::stringstream ss; ss<<msg;

//-------------------

#endif // PSALG_LOGGER_H

//-------------------

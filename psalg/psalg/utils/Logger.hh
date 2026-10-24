/**
 * @file
 * @brief Declares the psalg logging classes (Logger singleton, LogStream, LogRecord, LogFormatter, LogHandler, LogHandlerStdStreams) and the MSG, MSGLOG and MSGSTREAM macros.
 */
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
  std::string tstampNow(std::string fmt="%Y-%m-%d %H:%M:%S.f");
}
*/

//-------------------

namespace Logger {

//-------------------

class LogStream;
class LogRecord;
class LogHandler;
class LogHandlerStdStreams;

/**
 * Singleton logger (instance()) with a minimum level and a list of handlers; log() passes each record to every handler.
 * The instance starts at level INFO with one LogHandlerStdStreams.
 */
class Logger {

public:

  /** Message levels; logging(sev) is true when sev >= the current level. Note that DEBUG (0) is below TRACE (1). */
  enum LEVEL {DEBUG=0, /**< Value 0. */ TRACE, /**< Value 1. */ INFO, /**< Value 2. */ WARNING, /**< Value 3. */ ERROR, /**< Value 4. */ FATAL, /**< Value 5. */ NOLOG, /**< Value 6. */ LAST_LEVEL=NOLOG, /**< Equal to NOLOG (6). */ NUM_LEVELS=LAST_LEVEL+1 /**< Number of levels (7). */ };

  char* LEVELCN[NUM_LEVELS];  ///< Full level names ("DEBUG" ... "NOLOG"), filled by the constructor.
  char* LEVELC3[NUM_LEVELS];  ///< Three-letter level names ("DBG", "TRC", "INF", "WRN", "ERR", "FTL", "NLG"), returned by levelToName().
  char  LEVELC1[NUM_LEVELS];  ///< One-letter level codes ('D', 'T', 'I', 'W', 'E', 'F', 'N').

  /** Return the singleton, creating it on the first call (no locking). */
  static Logger* instance() {
    if(!_pinstance) _pinstance = new Logger();
    return _pinstance;
  }

  /** Increment the record counter and print the counter (4 digits), the 3-letter level and the text of ss to stdout; for levels above INFO also print the source file and line. */
  void logmsg(const LogStream& ss, const LEVEL& sev=DEBUG);

  /** Write the start timestamp, current level, log name and number of levels to out. */
  void loggerInfo(std::ostream& out);
  /** Return the timestamp taken when the logger was created (format %Y-%m-%d %H:%M:%S.%f, %f being milliseconds). */
  const std::string& tstampStart() const {return _tstamp_start;}

  /** Set the logger name shown by loggerInfo(). */
  inline void setLogname(const std::string& logname) {_logname=logname;}
  /** Set the minimum level used by logging(). */
  inline void setLevel(const LEVEL& level) {_level=level;}
  /** Return true if sev >= the current level. */
  inline bool logging(const LEVEL& sev) {return (sev >= _level);}
  /** Return the 3-letter name of level (e.g. "WRN"). */
  const char* levelToName(const LEVEL& level);
  /** Return the LEVEL for a full level name ("DEBUG", "TRACE", "INFO", "WARNING", "ERROR", "FATAL" or "NOLOG"); throws std::out_of_range for any other string. */
  LEVEL name_to_level(const std::string& name);
  /** Increment the record counter and pass rec to every handler. */
  void log(const LogRecord& rec);
  /** Return the number of records counted so far. */
  const unsigned counter() {return _counter;}
  /** Set the level and the time format of every handler's formatter. */
  void setLogger(const LEVEL& level=DEBUG, const std::string& timefmt="%H:%M:%S.%f");

  /// add a handler for the messages, takes ownership of the object
  void addHandler(LogHandler* handler) {_handlers.push_back(handler);}
  /** Set the time format of every handler's formatter (a strftime format in which %f is replaced by milliseconds). */
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

/** String stream that collects one message and, when destroyed, passes it to the singleton as a LogRecord. Used by the MSG, MSGLOG and MSGSTREAM macros. */
class LogStream : public std::stringstream {
public:

  /** Alias for Logger::LEVEL. */
  typedef Logger::LEVEL level_t;

  /** Store the logger name, level and source location; ok() starts as LOGGER.logging(sev). */
  LogStream(const std::string& logname, const level_t& sev, const char* file=0, int line=-1);
  /** Pass the collected text to LOGGER.log() as a LogRecord. This happens whether or not the level is enabled; the macros check the level before creating the stream. */
  virtual ~LogStream(){_emit_content();}
  /** Return this stream as a std::ostream reference. */
  std::ostream& logger_ostream() {return *this;}
  /** Return the source file name given to the constructor. */
  inline const char* file() const {return _filename;}
  /** Return the source line given to the constructor. */
  inline const int line() const {return _linenum;}

  /// get the state of the stream
  bool ok() const {return _ok;}

  // set the state of the stream to "not OK"
  /** Set ok() to false; MSGSTREAM uses this to end its one-pass loop. */
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

/** One log message: logger name (held by reference), level, source file and line, and a pointer to the stream buffer with the text. */
class LogRecord {

public:

  /** Alias for Logger::LEVEL. */
  typedef Logger::LEVEL level_t;

  /** Store the given values; nothing is copied (logger is held by reference, msgbuf as a pointer). */
  LogRecord(const std::string& logger,
            const level_t& level,
            const char* filename,
            int linenum,
            std::streambuf* msgbuf)
    : _logger(logger), _level(level), _filename(filename), _linenum(linenum), _msgbuf(msgbuf) {}

  /** Destructor; does nothing. */
  ~LogRecord() {}

  /// get logger name
  const std::string& logger() const {return _logger;}

  /// get message log level
  const level_t level() const {return _level;}

  /// get message location
  const char* file() const {return _filename;}
  /** Return the source line. */
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

/** Formats a LogRecord as: the time (if a time format is set), the 4-digit record counter, the 3-letter level, for levels other than INFO the file name (without directories), a colon and the line number, and the message text. */
class LogFormatter {

public:

  /** Alias for Logger::LEVEL. */
  typedef Logger::LEVEL level_t;

  /** Store timefmt; fmt is ignored. */
  LogFormatter(const std::string& fmt="", const std::string& timefmt=""); //%Y-%m-%d %H:%M:%S.%f");

  /** Destructor; does nothing. */
  virtual ~LogFormatter() {}

  /// add format
  virtual void addFormat(const level_t& level, const std::string& fmt);

  /** Set the strftime time format (%f is replaced by milliseconds); an empty format leaves the time out. */
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

/** Abstract handler that writes LogRecords with a LogFormatter it owns (a default one is created on first use if none was set). */
class LogHandler {

public:

  /** Delete the formatter. */
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

/** Handler that formats records of level INFO or lower to stdout and higher levels to stderr, each followed by std::endl. */
class LogHandlerStdStreams : public LogHandler {

public:
  /** Construct with no formatter set (a default one is created on first use). */
  LogHandlerStdStreams();
  /** Destructor; does nothing beyond the base class. */
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
/** Shortcut for Logger::Logger. */
#define LL Logger::Logger
/** The singleton logger, (*Logger::Logger::instance()). */
#define LOGGER (*LL::instance())
/** Expands to __FILE__,__LINE__. */
#define __MACROPARS __FILE__,__LINE__

#ifdef MSGLOG
#undef MSGLOG
#endif
/** If level sev is enabled, stream msg into a LogStream with logger name logger and the current file and line; it is logged when the statement ends. */
#define MSGLOG(logger,sev,msg) \
  if (LOGGER.logging(LL::sev)){ \
    Logger::LogStream _log_stream(logger,LL::sev,__MACROPARS); _log_stream.logger_ostream() << msg; \
  }

#ifdef MSG
#undef MSG
#endif
/** If level sev is enabled, stream msg into a LogStream with an empty logger name and the current file and line; it is logged when the statement ends. */
#define MSG(sev,msg) \
  if (LOGGER.logging(LL::sev)){ \
    Logger::LogStream _log_stream(std::string(),LL::sev,__MACROPARS); _log_stream << msg; \
  }

#ifdef MSGSTREAM
#undef MSGSTREAM
#endif
/** If level sev is enabled, run the following statement or block once with a LogStream named strm (empty logger name), then log what was written to it. */
#define MSGSTREAM(sev,strm) \
    if (LOGGER.logging(LL::sev)) \
      for(Logger::LogStream strm(std::string(),LL::sev,__MACROPARS); strm.ok(); strm.finish())

//-------------------

#endif // PSALG_LOGGER_H

//-------------------

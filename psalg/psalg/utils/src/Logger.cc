
//-------------------
#include "psalg/utils/Logger.hh" // MsgLog, Logger, LOGPRINT, LOGMSG
//-------------------
#include <iostream> // cout
#include <stdexcept>
#include <cstring>  // memcpy
#include <iomanip>   // std::setw(8), setfill, left

#include <time.h> // clock_gettime, localtime_r

namespace {
  // default format string
  const std::string s_def_fmt = "TBD default format";
  const std::string s_def_tst = "%Y-%m-%d %H:%M:%S.%f";

  // get current time and format it
  void formattedTime(std::string fmt, std::ostream& out)
  {
    // get seconds/nanoseconds
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    // convert to break-down time
    struct tm tms;
    localtime_r(&ts.tv_sec, &tms);

    // replace %f in the format string with miliseconds
    std::string::size_type n = fmt.find("%f");
    if (n != std::string::npos) {
      char subs[4];
      snprintf(subs, 4, "%03d", int(ts.tv_nsec/1000000));
      while(n != std::string::npos) {
	fmt.replace(n, 2, subs);
	n = fmt.find("%f");
      }
    }

    char buf[1024] ;
    strftime(buf, 1024, fmt.c_str(), &tms);
    out << buf ;
  }

//-------------------

  const char* tstampNow(std::string fmt) {
    std::stringstream ss; 
    formattedTime(fmt, ss);
    return ss.str().c_str(); 
  }

} // namespace unnamed

//-------------------
//-------------------
//-------------------

namespace Logger {

//-------------------

  Logger::Logger() : _counter(0), _logname(""), _level(Logger::INFO), _tstamp_start(tstampNow(s_def_tst)), _handlers() {
  _initLevelNames();
  addHandler(new LogHandlerStdStreams);
}

//-------------------

Logger::~Logger() {
  for (HandlerList::const_iterator it = _handlers.begin(); it != _handlers.end(); ++it) delete *it;
  delete _pinstance;
} 

//-------------------

void Logger::_initLevelNames() {
  char* levelcn[NUM_LEVELS]={(char*)"DEBUG", (char*)"TRACE", (char*)"INFO", (char*)"WARNING", (char*)"ERROR", (char*)"FATAL", (char*)"NOLOG"};
  char* levelc3[NUM_LEVELS]={(char*)"DBG", (char*)"TRC", (char*)"INF", (char*)"WRN", (char*)"ERR", (char*)"FTL", (char*)"NLG"};
  char  levelc1[NUM_LEVELS]={'D','T','I','W','E','F','N'};
  memcpy(LEVELCN, levelcn, sizeof(levelcn));
  memcpy(LEVELC3, levelc3, sizeof(levelc3));
  memcpy(LEVELC1, levelc1, sizeof(levelc1));
}

//-------------------

Logger* Logger::_pinstance = NULL; // init static pointer for singleton

//-------------------

void Logger::loggerInfo(std::ostream& out) {
  //formattedTime("%Y-%m-%d %H:%M:%S", out);
  out << tstampStart()
      << " Single instance of the class Logger:"
      << " level: " << levelToName(_level)
      << " logname: \"" << _logname << '\"'
      << " number of levels=" << NUM_LEVELS
      << '\n';
}

//-------------------

const char* Logger::levelToName(const Logger::LEVEL& level)
{
  return LEVELC3[level];
}

//-------------------

Logger::LEVEL Logger::name_to_level(const std::string& levname) {
  if      (levname == "TRACE")   return Logger::TRACE;
  else if (levname == "DEBUG" )  return Logger::DEBUG;
  else if (levname == "WARNING") return Logger::WARNING;
  else if (levname == "INFO")    return Logger::INFO;
  else if (levname == "ERROR")   return Logger::ERROR;
  else if (levname == "FATAL")   return Logger::FATAL;
  else if (levname == "NOLOG")   return Logger::NOLOG;
  else throw std::out_of_range("unexpected logging level name");
}

//-------------------

void Logger::logmsg(const LogStream& ss, const Logger::LEVEL& sev) {
  _counter++;
  std::cout << std::setfill('0') << std::setw(4)<< _counter <<  ' ' << LEVELC3[sev] << ' ' << ss.str();
  if(sev>Logger::INFO) std::cout << " from:" << ss.file() <<  " line:" << ss.line() << '\n';
  else std::cout << '\n';
} 

//-------------------

/// get the stream for the specified log level
void Logger::log(const LogRecord& rec) {
  _counter++;

  for (HandlerList::const_iterator it = _handlers.begin(); it != _handlers.end(); ++it)
      (*it)->log(rec);
}

//-------------------

void Logger::setTimeFormat(const std::string& timefmt) {
  for (HandlerList::const_iterator it = _handlers.begin(); it != _handlers.end(); ++it) 
      (*it)->formatter().setTimeFormat(timefmt);
}
//-------------------

void Logger::setLogger(const LEVEL& level, const std::string& timefmt) {
  setLevel(level);
  setTimeFormat(timefmt);
}

//-------------------
//-------------------
//-------------------

LogStream::LogStream(const std::string& logname, const Logger::LEVEL& sev, const char* file, int line)
   : std::stringstream(), _logname(logname), _sev(sev), _filename(file), _linenum(line) {
   _ok = LOGGER.logging(sev);
};

//-------------------

void LogStream::_emit_content() const {
  LogRecord record(_logname, _sev, _filename, _linenum, rdbuf());
  LOGGER.log(record);
}

//-------------------
//-------------------
//-------------------

LogFormatter::LogFormatter(const std::string& afmt, const std::string& timefmt)
  : _timefmt(timefmt)
{
  //if(_timefmt.empty()) _timefmt = "%H:%M:%S.%f"; // "%Y-%m-%d %H:%M:%S.%f";
}

// add level-specific format
void
LogFormatter::addFormat(const level_t& level, const std::string& fmt) {
  _fmtMap[level] = fmt;
}

// set time format
void
LogFormatter::setTimeFormat(const std::string& timefmt) {
  _timefmt = timefmt;
}

// get a format string for a given level
const std::string&
LogFormatter::getFormat(const level_t& level) const {
  if (not _fmtMap[level].empty()) return _fmtMap[level];
  else return s_def_fmt;// for level " + LOGGER.LEVELCN[level];
}

// format message to the output stream
void
LogFormatter::format(const LogRecord& rec, std::ostream& out)
{
  //out << "In LogFormatter";
  //const std::string& fmt = getFormat(rec.level());

  formattedTime(_timefmt, out);

  std::string s = rec.file();
  std::size_t pos = s.rfind('/');
  std::string str = (pos != std::string::npos) ? s.substr(pos+1) : s; 
  if(!_timefmt.empty()) out << ' '; 
  out << std::setfill('0') << std::setw(4) << LOGGER.counter()
      << ' ' << LOGGER.levelToName(rec.level());

  if(rec.level() != LL::INFO)
      out << ' ' << str << ':' << rec.line();

  out << ' ' << rec.msgbuf();
}

//-------------------
//-------------------
//-------------------

LogHandler::LogHandler() : _formatter(0) {}

LogHandler::~LogHandler(){delete _formatter;}

void
LogHandler::setFormatter(LogFormatter* formatter){delete _formatter; _formatter = formatter;}

// get the formatter
LogFormatter& 
LogHandler::formatter() const {
  if (! _formatter) _formatter = new LogFormatter;
  return *_formatter ;
}

//-------------------
//-------------------
//-------------------

LogHandlerStdStreams::LogHandlerStdStreams() : LogHandler() {}

LogHandlerStdStreams::~LogHandlerStdStreams(){}

/// get the stream for the specified log level
bool
LogHandlerStdStreams::log(const LogRecord& record) const
{
  //if(! logging(record.level())) return false;

  if (record.level() <= Logger::LEVEL::INFO) {
    formatter().format(record, std::cout) ;
    std::cout << std::endl;
  } else {
    formatter().format(record, std::cerr);
    std::cerr << std::endl;
  }

  return true ;
}


//-------------------
//-------------------

} // namespace Logger

//-------------------
//-------------------







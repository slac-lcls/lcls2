#ifndef Pds_Cphw_Logging_hh
#define Pds_Cphw_Logging_hh

namespace Pds {
  namespace Cphw {
    class Logger {
    public:
      static Logger& instance();
    public:
      void debug   (const char* fmt...);
      void info    (const char* fmt...);
      void warning (const char* fmt...);
      void error   (const char* fmt...);
      void critical(const char* fmt...);
    };
  }
}

#endif

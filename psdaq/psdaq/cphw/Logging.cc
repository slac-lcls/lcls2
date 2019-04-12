#include "psdaq/cphw/Logging.hh"

#include <stdio.h>

using namespace Pds::Cphw;

static Logger* _instance = 0;

Logger& Logger::instance() {
  if (_instance) return *_instance;
  return *(_instance = new Logger);
}

enum Level { _DEBUG, _INFO, _WARNING, _ERROR, _CRITICAL };

static const char* label[] = {"DEBUG","INFO","WARNING","ERROR","CRITICAL"};

#define handle(level)                          \
  {                                             \
  printf("%s: ", label[level]);                 \
  printf(fmt);                                  \
  printf("\n");                                 \
}

void Logger::debug   (const char* fmt...) { printf(fmt); }
void Logger::info    (const char* fmt...) { handle(_INFO    ); }
void Logger::warning (const char* fmt...) { handle(_WARNING ); }
void Logger::error   (const char* fmt...) { handle(_ERROR   ); }
void Logger::critical(const char* fmt...) { handle(_CRITICAL); }

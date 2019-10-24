#ifndef PDS_SYSLOG_HH
#define PDS_SYSLOG_HH

#include <syslog.h>     // defines LOG_WARNING, etc

#define SYSLOG_IDENT_MAX    32
#define SYSLOG_FORMAT_MAX   4096

namespace Pds {
class SysLog
{
public:
    // initialize
    static void init(const char *instrument, int level);

    // log
    static void debug(const char *fmt, ...);
    static void info(const char *fmt, ...);
    static void warning(const char *fmt, ...);
    static void error(const char *fmt, ...);
    static void critical(const char *fmt, ...);
};
}

#endif


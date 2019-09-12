#ifndef PDS_SYSLOG_HH
#define PDS_SYSLOG_HH

#include <syslog.h>
#include <stdarg.h>

namespace Pds {
class SysLog
{
public:
    SysLog(const char *instrument, int level);

    static void init(const char *instrument, int level) 
    {
        static char ident[20];
        if (instrument) {
            snprintf(ident, sizeof(ident)-1, "%s-%s", instrument, program_invocation_short_name);
        } else {
            snprintf(ident, sizeof(ident)-1, "%s", program_invocation_short_name);
        }
        openlog(ident, LOG_PERROR | LOG_PID, LOG_USER);
        setlogmask(LOG_UPTO(level));
    }

    static void debug(const char *fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        vsyslog(LOG_DEBUG, fmt, args);
        va_end(args);
    }

    static void info(const char *fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        vsyslog(LOG_INFO, fmt, args);
        va_end(args);
    }

    static void warning(const char *fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        vsyslog(LOG_WARNING, fmt, args);
        va_end(args);
    }

    static void error(const char *fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        vsyslog(LOG_ERR, fmt, args);
        va_end(args);
    }

    static void critical(const char *fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        vsyslog(LOG_CRIT, fmt, args);
        va_end(args);
    }
};
}

#endif


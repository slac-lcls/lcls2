#include <stdio.h>
#include <stdarg.h>
#include "psdaq/service/SysLog.hh"

extern char *program_invocation_short_name;

using namespace Pds;

void SysLog::init(const char *instrument, int level)
{
    static char ident[20];
    if (instrument) {
        snprintf(ident, sizeof(ident)-1, "%s-%s", instrument, program_invocation_short_name);
    } else {
        snprintf(ident, sizeof(ident)-1, "%s", program_invocation_short_name);
    }
    openlog(ident, LOG_PID | LOG_PERROR, LOG_USER);
    setlogmask(LOG_UPTO(level));
}

void SysLog::debug(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsyslog(LOG_DEBUG, fmt, args);
    va_end(args);
}

void SysLog::info(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsyslog(LOG_INFO, fmt, args);
    va_end(args);
}

void SysLog::warning(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsyslog(LOG_WARNING, fmt, args);
    va_end(args);
}

void SysLog::error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsyslog(LOG_ERR, fmt, args);
    va_end(args);
}

void SysLog::critical(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsyslog(LOG_CRIT, fmt, args);
    va_end(args);
}

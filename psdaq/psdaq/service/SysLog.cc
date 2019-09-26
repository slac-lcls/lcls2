#include <stdio.h>
#include <errno.h>
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
    openlog(ident, LOG_PID, LOG_USER);
    setlogmask(LOG_UPTO(level));
}

void SysLog::debug(const char *fmt, ...)
{
    va_list args;

    // syslog
    va_start(args, fmt);
    vsyslog(LOG_DEBUG, fmt, args);
    va_end(args);

    // stderr
    fprintf(stderr, "DEBUG ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void SysLog::info(const char *fmt, ...)
{
    va_list args;

    // syslog
    va_start(args, fmt);
    vsyslog(LOG_INFO, fmt, args);
    va_end(args);

    // stderr
    fprintf(stderr, "INFO ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void SysLog::warning(const char *fmt, ...)
{
    va_list args;

    // syslog
    va_start(args, fmt);
    vsyslog(LOG_WARNING, fmt, args);
    va_end(args);

    // stderr
    fprintf(stderr, "WARNING ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void SysLog::error(const char *fmt, ...)
{
    va_list args;

    // syslog
    va_start(args, fmt);
    vsyslog(LOG_ERR, fmt, args);
    va_end(args);

    // stderr
    fprintf(stderr, "ERROR ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void SysLog::critical(const char *fmt, ...)
{
    va_list args;

    // syslog
    va_start(args, fmt);
    vsyslog(LOG_CRIT, fmt, args);
    va_end(args);

    // stderr
    fprintf(stderr, "CRITICAL ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

#include <stdio.h>
#include <stdarg.h>
#include "psalg/utils/SysLog.hh"

extern char *program_invocation_short_name;

using namespace psalg;

void SysLog::init(const char *instrument, int level)
{
    static char ident[SYSLOG_IDENT_MAX];
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
    char newfmt[SYSLOG_FORMAT_MAX];
    va_list args;
    va_start(args, fmt);
    snprintf(newfmt, sizeof(newfmt), "<D> %s", fmt);
    vsyslog(LOG_DEBUG, newfmt, args);
    va_end(args);
}

void SysLog::info(const char *fmt, ...)
{
    char newfmt[SYSLOG_FORMAT_MAX];
    va_list args;
    va_start(args, fmt);
    snprintf(newfmt, sizeof(newfmt), "<I> %s", fmt);
    vsyslog(LOG_INFO, newfmt, args);
    va_end(args);
}

void SysLog::warning(const char *fmt, ...)
{
    char newfmt[SYSLOG_FORMAT_MAX];
    va_list args;
    va_start(args, fmt);
    snprintf(newfmt, sizeof(newfmt), "<W> %s", fmt);
    vsyslog(LOG_WARNING, newfmt, args);
    va_end(args);
}

void SysLog::error(const char *fmt, ...)
{
    char newfmt[SYSLOG_FORMAT_MAX];
    va_list args;
    va_start(args, fmt);
    snprintf(newfmt, sizeof(newfmt), "<E> %s", fmt);
    vsyslog(LOG_ERR, newfmt, args);
    va_end(args);
}

void SysLog::critical(const char *fmt, ...)
{
    char newfmt[SYSLOG_FORMAT_MAX];
    va_list args;
    va_start(args, fmt);
    snprintf(newfmt, sizeof(newfmt), "<C> %s", fmt);
    vsyslog(LOG_CRIT, newfmt, args);
    va_end(args);
}

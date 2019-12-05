#ifndef PDS_SYSLOG_HH
#define PDS_SYSLOG_HH

#include <stdio.h>
#include <stdarg.h>
#include <syslog.h>     // defines LOG_WARNING, etc

#undef GET_PROGRAM_NAME
#ifdef __GLIBC__
    extern "C" char *program_invocation_short_name;
#   define GET_PROGRAM_NAME() program_invocation_short_name
#else /* *BSD and OS X */
#   include <stdlib.h>
#   define GET_PROGRAM_NAME() getprogname()
#endif

#define SYSLOG_IDENT_MAX    32
#define SYSLOG_FORMAT_MAX   4096

namespace psalg {
    class SysLog {
        public:

        static void init(const char *instrument, int level)
        {
            static char ident[SYSLOG_IDENT_MAX];
            if (instrument) {
                snprintf(ident, sizeof(ident)-1, "%s-%s", instrument, GET_PROGRAM_NAME());
            } else {
                snprintf(ident, sizeof(ident)-1, "%s", GET_PROGRAM_NAME());
            }
            openlog(ident, LOG_PID | LOG_PERROR, LOG_USER);
            setlogmask(LOG_UPTO(level));
        }

        static void debug(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<D> %s", fmt);
            vsyslog(LOG_DEBUG, newfmt, args);
            va_end(args);
        }

        static void info(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<I> %s", fmt);
            vsyslog(LOG_INFO, newfmt, args);
            va_end(args);
        }

        static void warning(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<W> %s", fmt);
            vsyslog(LOG_WARNING, newfmt, args);
            va_end(args);
        }

        static void error(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<E> %s", fmt);
            vsyslog(LOG_ERR, newfmt, args);
            va_end(args);
        }

        static void critical(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<C> %s", fmt);
            vsyslog(LOG_CRIT, newfmt, args);
            va_end(args);
        }
    };
}

#endif

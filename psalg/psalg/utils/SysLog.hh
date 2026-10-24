/**
 * @file
 * @brief Declares psalg::SysLog, static wrappers around syslog(3) that tag messages by level.
 */
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
/** Program name used in the syslog ident: program_invocation_short_name with glibc, getprogname() otherwise (this line is the non-glibc branch). */
#   define GET_PROGRAM_NAME() getprogname()
#endif

/** Size (32) of the static ident buffer used by SysLog::init(). */
#define SYSLOG_IDENT_MAX    32
/** Size (4096) of the format buffer used by the SysLog message functions. */
#define SYSLOG_FORMAT_MAX   4096

namespace psalg {
    /** Static wrappers around syslog(3): init() opens the log, and each message function prefixes the format with a level letter (D, I, W, E or C) in angle brackets. */
    class SysLog {
        public:

        /**
         * Open syslog (facility LOG_USER, options LOG_PID and LOG_PERROR, so messages also go to stderr) with ident instrument-program, or just the program name if instrument is null, truncated to 30 characters.
         * Then set the log mask to LOG_UPTO(level).
         */
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

        /** Log fmt with its arguments at LOG_DEBUG; the format is prefixed with the letter D in angle brackets and a space. */
        static void debug(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<D> %s", fmt);
            vsyslog(LOG_DEBUG, newfmt, args);
            va_end(args);
        }

        /** Log fmt with its arguments at LOG_INFO; the format is prefixed with the letter I in angle brackets and a space. */
        static void info(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<I> %s", fmt);
            vsyslog(LOG_INFO, newfmt, args);
            va_end(args);
        }

        /** Log fmt with its arguments at LOG_WARNING; the format is prefixed with the letter W in angle brackets and a space. */
        static void warning(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<W> %s", fmt);
            vsyslog(LOG_WARNING, newfmt, args);
            va_end(args);
        }

        /** Log fmt with its arguments at LOG_ERR; the format is prefixed with the letter E in angle brackets and a space. */
        static void error(const char *fmt, ...)
        {
            char newfmt[SYSLOG_FORMAT_MAX];
            va_list args;
            va_start(args, fmt);
            snprintf(newfmt, sizeof(newfmt), "<E> %s", fmt);
            vsyslog(LOG_ERR, newfmt, args);
            va_end(args);
        }

        /** Log fmt with its arguments at LOG_CRIT; the format is prefixed with the letter C in angle brackets and a space. */
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

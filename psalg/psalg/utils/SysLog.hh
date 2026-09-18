#ifndef PDS_SYSLOG_HH
#define PDS_SYSLOG_HH

#include <stdio.h>
#include <stdarg.h>
#include <syslog.h>     // defines LOG_WARNING, etc
#include <time.h>       // clock_gettime, localtime_r, strftime

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
#define SYSLOG_TSTAMP_MAX   40

namespace psalg {
    class SysLog {
        public:

        //  Format the current local time as "%Y-%m-%d %H:%M:%S.%<microseconds>",
        //  matching the format string used by psalg/utils/src/Logger.cc:16
        //  ("%Y-%m-%d %H:%M:%S.%f") but with microsecond rather than
        //  millisecond resolution.
        static void timestamp(char *buf, size_t size)
        {
            struct timespec ts;
            if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
                snprintf(buf, size, "0000-00-00 00:00:00.000000");
                return;
            }
            struct tm tms;
            localtime_r(&ts.tv_sec, &tms);
            char secs[SYSLOG_TSTAMP_MAX];
            if (strftime(secs, sizeof(secs), "%Y-%m-%d %H:%M:%S", &tms) == 0) {
                snprintf(buf, size, "0000-00-00 00:00:00.000000");
                return;
            }
            snprintf(buf, size, "%s.%06d", secs, int(ts.tv_nsec / 1000));
        }

        //  Expand `fmt` once into a buffer, echo it to stderr with a leading
        //  timestamp, then hand the same expanded text to syslog().  This
        //  replaces syslog's LOG_PERROR echo, which is emitted raw with no
        //  timestamp of its own.  The syslogd-side record is deliberately NOT
        //  timestamped here, so it is not double-timestamped by the daemon.
        //
        //  `args` is consumed exactly once, by vsnprintf.  syslog() is then
        //  called with a literal "%s" format so that any '%' characters in the
        //  expanded message are not re-interpreted as conversions.
        static void emit(int priority, const char *tag, const char *fmt, va_list args)
        {
            char msg[SYSLOG_FORMAT_MAX];
            int n = snprintf(msg, sizeof(msg), "%s ", tag);
            if (n < 0)  n = 0;
            if ((size_t)n < sizeof(msg))
                vsnprintf(msg + n, sizeof(msg) - n, fmt, args);

            //  setlogmask(0) reports the current mask without modifying it, so
            //  the stderr echo honors the level set by init() just as
            //  LOG_PERROR did.
            if (setlogmask(0) & LOG_MASK(priority)) {
                char tstamp[SYSLOG_TSTAMP_MAX];
                timestamp(tstamp, sizeof(tstamp));
                fprintf(stderr, "%s %s\n", tstamp, msg);
            }
            syslog(priority, "%s", msg);
        }

        static void init(const char *instrument, int level)
        {
            static char ident[SYSLOG_IDENT_MAX];
            if (instrument) {
                snprintf(ident, sizeof(ident)-1, "%s-%s", instrument, GET_PROGRAM_NAME());
            } else {
                snprintf(ident, sizeof(ident)-1, "%s", GET_PROGRAM_NAME());
            }
            //  LOG_PERROR is deliberately NOT used: its stderr echo carries no
            //  timestamp.  Each level method below echoes to stderr itself.
            openlog(ident, LOG_PID, LOG_USER);
            setlogmask(LOG_UPTO(level));
        }

        static void debug(const char *fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            emit(LOG_DEBUG, "<D>", fmt, args);
            va_end(args);
        }

        static void info(const char *fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            emit(LOG_INFO, "<I>", fmt, args);
            va_end(args);
        }

        static void warning(const char *fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            emit(LOG_WARNING, "<W>", fmt, args);
            va_end(args);
        }

        static void error(const char *fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            emit(LOG_ERR, "<E>", fmt, args);
            va_end(args);
        }

        static void critical(const char *fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            emit(LOG_CRIT, "<C>", fmt, args);
            va_end(args);
        }
    };
}

#endif

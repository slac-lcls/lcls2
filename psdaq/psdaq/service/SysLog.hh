#ifndef PDS_SYSLOG_HH
#define PDS_SYSLOG_HH

#include <syslog.h>

namespace Pds {
class SysLog
{
public:
    SysLog(const char *experiment, int level);
    ~SysLog();
};
}

// Support same subset of syslog severities as the
// Python standard library logging module.
#define logCritical(...)    syslog(LOG_CRIT, __VA_ARGS__)
#define logError(...)       syslog(LOG_ERR, __VA_ARGS__)
#define logWarning(...)     syslog(LOG_WARNING, __VA_ARGS__)
#define logInfo(...)        syslog(LOG_INFO, __VA_ARGS__)
#define logDebug(...)       syslog(LOG_DEBUG, __VA_ARGS__)

#endif


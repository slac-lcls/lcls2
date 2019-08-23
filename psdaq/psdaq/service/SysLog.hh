#ifndef PDS_SYSLOG_HH
#define PDS_SYSLOG_HH

#include <syslog.h>

namespace Pds {
class SysLog
{
public:
    SysLog(const char *ident);
    void logDebug(const char *msg);
    void logInfo(const char *msg);
    void logWarning(const char *msg);
    void logError(const char *msg);
};
}

#endif


#ifndef PDS_SYSLOG_HH
#define PDS_SYSLOG_HH

#include <syslog.h>

namespace Pds {
class SysLog
{
public:
    SysLog(const char *ident);
    void debug(const char *msg);
    void info(const char *msg);
    void warning(const char *msg);
    void error(const char *msg);
    void alert(const char *msg);
};
}

#endif


#include "psdaq/service/SysLog.hh"

using namespace Pds;

SysLog::SysLog(const char *ident) 
{
    openlog(ident, LOG_PERROR | LOG_PID, LOG_LOCAL0);
}

void SysLog::logDebug(const char *msg)
{
    syslog(LOG_DEBUG, "%s", msg);
}

void SysLog::logInfo(const char *msg)
{
    syslog(LOG_INFO, "%s", msg);
}

void SysLog::logWarning(const char *msg)
{
    syslog(LOG_WARNING, "%s", msg);
}

void SysLog::logError(const char *msg)
{
    syslog(LOG_ERR, "%s", msg);
}

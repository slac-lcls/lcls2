#include "psdaq/service/SysLog.hh"

using namespace Pds;

SysLog::SysLog(const char *ident) 
{
    openlog(ident, LOG_PERROR | LOG_PID, LOG_LOCAL0);
}

void SysLog::debug(const char *msg)
{
    syslog(LOG_DEBUG, "%s", msg);
}

void SysLog::info(const char *msg)
{
    syslog(LOG_INFO, "%s", msg);
}

void SysLog::warning(const char *msg)
{
    syslog(LOG_WARNING, "%s", msg);
}

void SysLog::error(const char *msg)
{
    syslog(LOG_ERR, "%s", msg);
}

void SysLog::alert(const char *msg)
{
    syslog(LOG_ALERT, "%s", msg);
}

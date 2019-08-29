#include <stdio.h>
#include <errno.h>
#include "psdaq/service/SysLog.hh"

extern char *program_invocation_short_name;

using namespace Pds;

SysLog::SysLog(const char *experiment, int level) 
{
    static char ident[20];
    snprintf(ident, sizeof(ident)-1, "%s-%s", experiment, program_invocation_short_name);
    openlog(ident, LOG_PERROR | LOG_PID, LOG_USER);
    setlogmask(LOG_UPTO(level));
}

SysLog::~SysLog() 
{
    closelog();
}

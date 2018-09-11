#include "XtcMonitorMsg.hh"

#include <stdio.h>

using namespace Pds::MonReq;

void XtcMonitorMsg::sharedMemoryName     (const char* tag, char* buffer)
{
  sprintf(buffer,"/PdsMonitorSharedMemory_%s",tag);
}

void XtcMonitorMsg::eventInputQueue      (const char* tag, unsigned client, char* buffer)
{
  sprintf(buffer,"/PdsToMonitorEvQueue_%s_%d",tag,client);
}

void XtcMonitorMsg::eventOutputQueue     (const char* tag, unsigned client, char* buffer)
{
  sprintf(buffer,"/PdsToMonitorEvQueue_%s_%d",tag,client+1);
}

void XtcMonitorMsg::transitionInputQueue (const char* tag, unsigned client, char* buffer)
{
  sprintf(buffer,"/PdsToMonitorTrQueue_%s_%d",tag,client);
}

void XtcMonitorMsg::discoveryQueue       (const char* tag, char* buffer)
{
  sprintf(buffer,"/PdsFromMonitorDiscovery_%s",tag);
}

void XtcMonitorMsg::registerQueue        (const char* tag, char* buffer, int id)
{
  sprintf(buffer,"/PdsToMonitorDiscovery_%s_%d",tag,id);
}


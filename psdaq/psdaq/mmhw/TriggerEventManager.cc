#include "psdaq/mmhw/TriggerEventManager.hh"

using namespace Pds::Mmhw;

void TriggerEventBuffer::start (unsigned group,
                                unsigned triggerDelay,
                                unsigned pauseThresh)
{
  this->resetCounters= 1;
  this->group        = group;
  this->pauseThresh  = pauseThresh;
  this->triggerDelay = triggerDelay;
  this->resetCounters= 0;
  this->enable       = 3;  // b0 = enable triggers, b1 = enable axiStream
}

void TriggerEventBuffer::stop  ()
{
  this->enable       = 0;
}

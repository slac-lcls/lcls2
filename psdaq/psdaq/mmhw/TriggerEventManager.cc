#include "psdaq/mmhw/TriggerEventManager.hh"

using namespace Pds::Mmhw;

void TriggerEventBuffer::start (unsigned group,
                                unsigned triggerDelay,
                                unsigned pauseThresh)
{
  volatile uint32_t& reset = *reinterpret_cast<volatile uint32_t*>(&this->resetCounters);
  reset= 1;
  this->group        = group;
  this->pauseThresh  = pauseThresh;
  this->triggerDelay = triggerDelay;
  reset = 0;
  this->enable       = 3;  // b0 = enable triggers, b1 = enable axiStream
}

void TriggerEventBuffer::stop  ()
{
  this->enable       = 0;
}

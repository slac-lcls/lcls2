#include "psdaq/service/SemLock.hh"

Pds::SemLock::SemLock() : _sem(Semaphore::EMPTY)
{
}

void Pds::SemLock::lock() { _sem.take(); }

void Pds::SemLock::release() { _sem.give(); }

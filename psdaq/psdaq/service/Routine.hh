#ifndef ODF_ROUTINE_H
#define ODF_ROUTINE_H

#include "Queue.hh"

/*
 * Functor specification for jobs that can be sent to the task
 * on its processing queue. The call() method of Task allows 
 * inserting anything derived from Routine to be put on the 
 * queue.
 */
namespace Pds {
class Routine : public Entry {
 public:
  virtual ~Routine() {}
  virtual void routine(void) = 0;
};
}
#endif

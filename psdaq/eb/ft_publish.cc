#include "psdaq/eb/Endpoint.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace Pds::Fabrics;
using namespace Pds;

class Listener : public Routine {
  public:
    Listener(PassiveEndpoint* pendp, Semaphore& sem, Task* task, std::vector<Endpoint*>& subs) :
      _pendp(pendp),
      _sem(sem),
      _task(task),
      _subs(subs),
      _run(true)
  {
    _run = _pendp->listen();
    printf("listening\n");
  }

  ~Listener() {
    _sem.take();
    _run = false;
    _sem.give();
    delete _pendp;
  }

  void start() {
    _task->call(this);
  }

  void routine() {
    if (_run) {
      Endpoint *endp = _pendp->accept();

      if (endp) {
        printf("client connected!\n");
        _sem.take();
        _subs.push_back(endp);
        _sem.give();
      }
      _task->call(this);
    }
  }

  private:
    PassiveEndpoint*        _pendp;
    Semaphore&              _sem;
    Task*                   _task;
    std::vector<Endpoint*>& _subs;
    bool                    _run;
};

int main(int argc, char *argv[])
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s count\n", argv[0]);
    return -1;
  }

  unsigned buff_num = 10;
  size_t buff_size = sizeof(uint64_t)*buff_num;
  char* buff = new char[buff_size];
  uint64_t* data_buff = (uint64_t*) buff;
  uint64_t max_count = strtoul(argv[1], NULL, 0);
  
  
  data_buff[1] = 0xadd;
  data_buff[2] = 0xdeadbeef;
  for (unsigned i=3; i< buff_num; i++)
    data_buff[i] = i;

  PassiveEndpoint* pendp = new PassiveEndpoint(NULL, "1234");
  if (pendp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", pendp->error());
    return pendp->error_num();
  }

  // get a pointer to the fabric
  Fabric* fab = pendp->fabric();
  // register the memory buffer
  MemoryRegion* mr = fab->register_memory(buff, buff_size);
  if (!mr) {
    fprintf(stderr, "Failed to register memory region: %s\n", fab->error());
    return fab->error_num();
  }

  Semaphore sem(Semaphore::FULL);
  std::vector<Endpoint*> subs;
  Task* task = new Task(TaskObject("Sublisten",35));

  Listener* listener = new Listener(pendp, sem, task, subs);
  listener->start();

  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  for (uint64_t count=0; count < max_count; count++) {
    int num_comp;
    struct fi_cq_data_entry comp;
    data_buff[0] = count;
    for (unsigned i=3; i< buff_num; i++)
      data_buff[i]++;
    printf("sending data %lu out of %lu\n", count+1, max_count);
    sem.take();
    for (unsigned i=0; i<subs.size(); i++) {
      if (!subs[i]) continue;

      // check for shutdown events on the eq - if there are close endpoint and zero it on sub list
      if(subs[i]->event(&event, &entry, &cm_entry)) {
        if (cm_entry && event == FI_SHUTDOWN) {
          pendp->close(subs[i]);
          subs[i]=0;
          printf("client disconnected!\n");
          continue;
        }
      }

      // post a send for this sub
      if (!subs[i]->send(buff, buff_size, &count, mr)) {
        pendp->close(subs[i]);
        subs[i]=0;
        printf("error posting send to client %u - dropping!\n", i);
      }
    }
    for (unsigned i=0; i<subs.size(); i++) {
      if (!subs[i]) continue;

      // check if send has completed
      if (!subs[i]->comp_wait(&comp, &num_comp, 1, 1000)) {
        pendp->close(subs[i]);
        subs[i]=0;
        printf("error completing send to client %u - dropping!\n", i);
      }
    }
    sem.give();
    sleep(2);
  }

  delete listener;
  delete[] buff;

  return 0;
}

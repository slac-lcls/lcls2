#include "psdaq/eb/Endpoint.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <vector>

using namespace Pds::Fabrics;
using namespace Pds;

static const char* PORT_DEF = "12345";
static const uint64_t COUNT_DEF = 100;

static void showUsage(const char* p)
{
  printf("\nSimple libfabrics publisher example (its pair is the ft_sub example).\n"
         "\n"
         "This example is simulating something similar to zeromq's publish/subscribe model - a.k.a\n"
         "a simulated multicast using an underlying connected interface. The server periodically sends a\n"
         "data buffer to all of its subscribers. A client connects to server to 'subscribe' and disconnects\n"
         "to unsubscribe.\n"
         "\n"
         "Usage: %s [-h|--help]\n"
         "    -a|--addr     the local address to which the server binds (default: libfabrics 'best' choice)\n"
         "    -p|--port     the port or libfaric 'service' the server will use (default: %s)\n"
         "    -c|--count    the max number of times to publish the data to subscribers before exiting (default: %lu)\n"
         "    -h|--help     print this message and exit\n", p, PORT_DEF, COUNT_DEF);
}

class Listener : public Routine {
  public:
    Listener(PassiveEndpoint* pendp, CompletionPoller* cpoller, Semaphore& sem, Task* task, std::vector<Endpoint*>& subs) :
      _pendp(pendp),
      _cpoller(cpoller),
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
        _cpoller->add(endp, endp->txcq());
        _sem.give();
      }
      _task->call(this);
    }
  }

  private:
    PassiveEndpoint*        _pendp;
    CompletionPoller*       _cpoller;
    Semaphore&              _sem;
    Task*                   _task;
    std::vector<Endpoint*>& _subs;
    bool                    _run;
};

int main(int argc, char *argv[])
{
  unsigned buff_num = 10;
  bool show_usage = false;
  const char *addr = NULL;
  const char *port = PORT_DEF;
  size_t buff_size = sizeof(uint64_t)*buff_num;
  char* buff = new char[buff_size];
  uint64_t* data_buff = (uint64_t*) buff;
  uint64_t max_count = COUNT_DEF;

  const char* str_opts = ":ha:p:c:";
  const struct option lo_opts[] =
  {
      {"help",  0, 0, 'h'},
      {"addr",  1, 0, 'a'},
      {"port",  1, 0, 'p'},
      {"count", 1, 0, 'c'},
      {0,       0, 0,  0 }
  };

  int option_idx = 0;
  while (int opt = getopt_long(argc, argv, str_opts, lo_opts, &option_idx )) {
    if ( opt == -1 ) break;

    switch(opt) {
      case 'h':
        showUsage(argv[0]);
        return 0;
      case 'a':
        addr = optarg;
        break;
      case 'p':
        port = optarg;
        break;
      case 'c':
        max_count = strtoul(optarg, NULL, 0);
        break;
      default:
        show_usage = true;
        break;
    }
  }

  if (optind < argc) {
    printf("%s: invalid argument -- %s\n", argv[0], argv[optind]);
    show_usage = true;
  }

  if (show_usage) {
    showUsage(argv[0]);
    return 1;
  }

  data_buff[1] = 0xadd;
  data_buff[2] = 0xdeadbeef;
  for (unsigned i=3; i< buff_num; i++)
    data_buff[i] = i;

  PassiveEndpoint* pendp = new PassiveEndpoint(addr, port);
  if (pendp->state() != EP_UP) {
    fprintf(stderr, "Failed to initialize fabrics endpoint: %s\n", pendp->error());
    return pendp->error_num();
  }

  // get a pointer to the fabric
  Fabric* fab = pendp->fabric();
  printf("using %s provider\n", fab->provider());
  // register the memory buffer
  MemoryRegion* mr = fab->register_memory(buff, buff_size);
  if (!mr) {
    fprintf(stderr, "Failed to register memory region: %s\n", fab->error());
    return fab->error_num();
  }
  // Create a cq poller
  CompletionPoller* cqpoll = new CompletionPoller(fab);

  Semaphore sem(Semaphore::FULL);
  std::vector<Endpoint*> subs;
  Task* task = new Task(TaskObject("Sublisten",35));

  Listener* listener = new Listener(pendp, cqpoll, sem, task, subs);
  listener->start();

  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  for (uint64_t count=0; count < max_count; count++) {
    int npend = 0;
    int num_comp;
    struct fi_cq_data_entry comp;
    bool dead = false;
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
          cqpoll->del(subs[i]);
          subs[i]=0;
          printf("client disconnected!\n");
          continue;
        }
      }

      // post a send for this sub
      if (subs[i]->send(buff, buff_size, &count, mr)) {
        pendp->close(subs[i]);
        cqpoll->del(subs[i]);
        subs[i]=0;
        printf("error posting send to client %u - dropping!\n", i);
      }

      npend++;
    }

    while (npend > 0) {
      if (cqpoll->poll()) {
        for (unsigned i=0; i<subs.size(); i++) {
          if (!subs[i]) continue;

          // check if send has completed
          if ( (num_comp = subs[i]->txcq()->comp(&comp, 1)) < 0) {
            if (subs[i]->error_num() != -FI_EAGAIN) {
              pendp->close(subs[i]);
              cqpoll->del(subs[i]);
              subs[i]=0;
              printf("error completing send to client %u - dropping!\n", i);
            }
          } else {
            npend -= num_comp;
          }
        }
      } else {
        printf("error polling completion queues: %s - aborting!", cqpoll->error());
        dead = true;
        break;
      }
    }

    sem.give();
    if (dead) break;
    sleep(2);
  }

  delete listener;
  delete[] buff;

  return 0;
}

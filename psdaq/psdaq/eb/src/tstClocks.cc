#include <chrono>
#include <string>
#include <iostream>
#include <math.h>
#include <unistd.h>

using ms_t = std::chrono::milliseconds;

class NLTimerScoped {
private:
  const std::chrono::steady_clock::time_point start;
  const std::string name;
  const unsigned count;

public:
  NLTimerScoped(const std::string& name, unsigned count=1) :
    start(std::chrono::steady_clock::now()),
    name (name),
    count(count)
  {
  }


  ~NLTimerScoped() {
    const auto end(std::chrono::steady_clock::now());
    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count();

    std::cout << name << " duration: " << double(duration_ns)/double(count) << " ns/iteration" << std::endl;
  }

};

#include "psdaq/service/fast_monotonic_clock.hh"

int main(int argc, const char * argv[]) {

  printf("fast_monotonic_clock resolution: %lu ns\n", Pds::fast_monotonic_clock::get_resolution().count());
  //printf("system_clock resolution: %lu ns\n", std::chrono::system_clock::get_resolution().count());
  //printf("steady_clock resolution: %lu ns\n", std::chrono::steady_clock::get_resolution().count());
  struct timespec t;
  auto result = clock_getres(CLOCK_MONOTONIC, &t);
  if (result == 0)
    printf("CLOCK_MONOTONIC resolution: %ld s, %ld ns\n", t.tv_sec, t.tv_nsec);
  else
    perror("CLOCK_MONOTONIC");
  result = clock_getres(CLOCK_MONOTONIC_COARSE, &t);
  if (result == 0)
    printf("CLOCK_MONOTONIC_COARSE resolution: %ld s, %ld ns\n", t.tv_sec, t.tv_nsec);
  else
    perror("CLOCK_MONOTONIC_COARSE");
  result = clock_getres(CLOCK_MONOTONIC_RAW, &t);
  if (result == 0)
    printf("CLOCK_MONOTONIC_RAW resolution: %ld s, %ld ns\n", t.tv_sec, t.tv_nsec);
  else
    perror("CLOCK_MONOTONIC_RAW");

  unsigned nIterations = 1000000;

  {
    NLTimerScoped timer( "1M * fast_monotonic_clock::now()", nIterations );

    for ( unsigned i=0; i < nIterations ; i++ ) {
      volatile auto t1(Pds::fast_monotonic_clock::now());
    }
  }

  {
    NLTimerScoped timer( "1M * fast_monotonic_clock::now(CLOCK_MONOTONIC_COARSE)", nIterations );

    for ( unsigned i=0; i < nIterations ; i++ ) {
      volatile auto t1(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_COARSE));
    }
  }

  {
    NLTimerScoped timer( "1M * fast_monotonic_clock::now(CLOCK_MONOTONIC)", nIterations );

    for ( unsigned i=0; i < nIterations ; i++ ) {
      volatile auto t1(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC));
    }
  }

  {
    NLTimerScoped timer( "1M * system_clock::now()", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      volatile auto t1(std::chrono::system_clock::now());
    }
  }

  {
    NLTimerScoped timer( "1M * steady_clock::now()", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      volatile auto t1(std::chrono::steady_clock::now());
    }
  }

  {
    NLTimerScoped timer( "1M * high_resolution_clock::now()", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      volatile auto t1(std::chrono::high_resolution_clock::now());
    }
  }

  {
    NLTimerScoped timer( "1M * MONOTONIC gettime", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      //timespec t;
      //clock_gettime(CLOCK_MONOTONIC, &t);
      volatile auto t1(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC));
    }
  }

  {
    NLTimerScoped timer( "1M * MONOTONIC_COARSE gettime", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      //timespec t;
      //clock_gettime(CLOCK_MONOTONIC_COARSE, &t);
      volatile auto t1(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_COARSE));
    }
  }

  {
    NLTimerScoped timer( "1M * MONOTONIC_RAW gettime", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      //timespec t;
      //clock_gettime(CLOCK_MONOTONIC_RAW, &t);
      volatile auto t1(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_RAW));
    }
  }

  nIterations = 100;

  {
    const ms_t    timeout(100);
    NLTimerScoped timer  ( "100 * fast_monotonic_clock 100 ms timeout", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      auto t0(Pds::fast_monotonic_clock::now());
      while (true)
      {
        auto t1(Pds::fast_monotonic_clock::now());
        if (t1 - t0 > timeout)  break;
      }
    }
  }

  {
    const ms_t    timeout(100);
    NLTimerScoped timer  ( "100 * system_clock 100 ms timeout", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      auto t0(std::chrono::system_clock::now());
      while (true)
      {
        auto t1(std::chrono::system_clock::now());
        if (t1 - t0 > timeout)  break;
      }
    }
  }

  {
    const ms_t    timeout(100);
    NLTimerScoped timer  ( "100 * steady_clock 100 ms timeout", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      auto t0(std::chrono::steady_clock::now());
      while (true)
      {
        auto t1(std::chrono::steady_clock::now());
        if (t1 - t0 > timeout)  break;
      }
    }
  }

  {
    const ms_t    timeout(100);
    NLTimerScoped timer  ( "100 * high_resolution_clock 100 ms timeout", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      auto t0(std::chrono::high_resolution_clock::now());
      while (true)
      {
        auto t1(std::chrono::high_resolution_clock::now());
        if (t1 - t0 > timeout)  break;
      }
    }
  }

  {
    const ms_t    timeout(100);
    NLTimerScoped timer  ( "100 * CLOCK_MONOTONIC 100 ms timeout", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      auto t0(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC));
      while (true)
      {
        auto t1(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC));
        if (t1 - t0 > timeout)  break;
      }
    }
  }

  {
    const ms_t    timeout(100);
    auto          t      (Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_COARSE));
    NLTimerScoped timer  ( "100 * CLOCK_MONOTONIC_COARSE 100 ms timeout", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      auto t0(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_COARSE));
      while (true)
      {
        auto t1(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_COARSE));
        if (t1 - t0 > timeout)  break;
      }
    }
  }

  {
    const ms_t    timeout(100);
    NLTimerScoped timer  ( "100 * CLOCK_MONOTONIC_RAW 100 ms timeout", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      auto t0(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_RAW));
      while (true)
      {
        auto t1(Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC_RAW));
        if (t1 - t0 > timeout)  break;
      }
    }
  }

  {
    NLTimerScoped timer( "100 * usleep(100000)", nIterations );

    for ( unsigned i=0; i < nIterations; i++ ) {
      usleep(100000);
    }
  }


  {
    NLTimerScoped timer( "sleep( 4 )" );

    sleep( 4 );
  }
}

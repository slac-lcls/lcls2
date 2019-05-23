#include "IndexPool.hh"

#include <cstdint>
#include <stdio.h>

typedef struct
{
  unsigned a;
  void*    b;
  uint64_t c;
  unsigned d[];
} MyStruct;

using namespace Pds::Eb;


int main(int argc, char **argv)
{
  unsigned count = 16;
  unsigned n     = 3;
  IndexPool<MyStruct>& pool = *new IndexPool<MyStruct>(count, n * sizeof(*MyStruct::d));

  printf("sizeof(pool) = %zd\n", sizeof(pool));

  pool.dump();

  printf("1: pool.size() = %zd\n", pool.size());
  printf("sizeof(MyStruct) + %d * sizeof(*MyStruct::d) = %zd == %zd ?\n",
         n, sizeof(MyStruct) + n * sizeof(*MyStruct::d), pool.sizeofObject());

  for (unsigned i = 0; i < 17; ++i)
  {
    printf("i = %d, p = %p\n", i, pool.allocate(i));
  }

  printf("2: pool.size() = %zd\n", pool.size());

  for (unsigned i = 0; i < 17; ++i)
  {
    MyStruct& p = pool[i];
    printf("&pool[%d] = %p, allocated = %c, idx = %d\n",
           i, &p, pool.isAllocated(i) ? 'Y' : 'N', pool.index(&p));
    pool.free(&p);
  }

  printf("3: pool.size() = %zd\n", pool.size());

  pool.dump();

  delete &pool;

  return 0;
}

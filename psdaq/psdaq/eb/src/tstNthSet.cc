#ifdef NDEBUG
# undef NDEBUG
#endif

#include <cassert>
#include <cstdint>
#include <x86intrin.h>
#include <stdio.h>

inline uint64_t nthset(uint64_t x, unsigned n) {
  return _pdep_u64(1ULL << n, x);
}

int main() {
  for (unsigned i = 0; i < 64; ++i)
  {
    uint64_t x = -1ul; //1ul << i;
    uint64_t v = nthset(x, i); //0);
    int      b = __builtin_ffsl(v) - 1;
    printf("i = %d, x = %016lx, v = %016lx, Nth bit = %d\n", i, x, v, b);
  }

  printf("%08lx, %08lx, %08lx\n",
         0b00001101100001001100100010100000ul,
         0b00000000000000000000000000100000ul,
         nthset(0b00001101100001001100100010100000, 0));

  printf("%08lx, %08lx, %08llx\n",
         0xdeadbeeful, 0xff000000ul, _pdep_u64(0xdeadbeef, 0xff000000));
  printf("%08lx, %08lx, %08llx\n",
         0xdeadbeeful, 0x00fff000ul, _pdep_u64(0xdeadbeef, 0x00fff000));
  printf("%08lx, %08lx, %08llx\n",
         0xdeadbeeful, 0x00000ffful, _pdep_u64(0xdeadbeef, 0x00000fff));

  assert(nthset(0b00001101100001001100100010100000, 0) ==
         0b00000000000000000000000000100000);
  assert(nthset(0b00001101100001001100100010100000, 1) ==
         0b00000000000000000000000010000000);
  assert(nthset(0b00001101100001001100100010100000, 3) ==
         0b00000000000000000100000000000000);
  assert(nthset(0b00001101100001001100100010100000, 9) ==
         0b00001000000000000000000000000000);
  assert(nthset(0b00001101100001001100100010100000, 10) ==
         0b00000000000000000000000000000000);
}

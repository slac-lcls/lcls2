#ifndef Psdaq_Cphw_Utils_hh
#define Psdaq_Cphw_Utils_hh

static inline unsigned getf(unsigned i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return n > 31 ? v : (v>>sh)&((1<<n)-1);
}

static inline unsigned getf(const Pds::Cphw::Reg& i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return n > 31 ? v : (v>>sh)&((1<<n)-1);
}

static inline unsigned setf(Pds::Cphw::Reg& o, unsigned v, unsigned n, unsigned sh)
{
  if (n>31) {
    o = v;
    return v;
  }
  unsigned r = unsigned(o);
  unsigned q = r;
  q &= ~(((1<<n)-1)<<sh);
  q |= (v&((1<<n)-1))<<sh;
  o = q;

  /*
  if (q != unsigned(o)) {
    printf("setf[%p] failed: %08x != %08x\n", &o, unsigned(o), q);
  }
  else if (q != r) {
    printf("setf[%p] passed: %08x [%08x]\n", &o, q, r);
  }
  */
  return q;
}


#endif

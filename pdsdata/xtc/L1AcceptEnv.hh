#ifndef Pds_L1AcceptEnv_hh
#define Pds_L1AcceptEnv_hh

#include "pdsdata/xtc/Env.hh"

namespace Pds {
  class L1AcceptEnv : public Env {
  public:
    enum { MaxReadoutGroups = 8 };
    enum L3TResult { None, Pass, Fail };
  public:  
    L1AcceptEnv();
    L1AcceptEnv(unsigned groups);
    L1AcceptEnv(unsigned groups, L3TResult l3t);
    L1AcceptEnv(unsigned groups, L3TResult l3t, bool trim, bool unbiased);
  public:
    uint32_t  clientGroupMask() const;
    L3TResult l3t_result     () const;
    bool      trimmed        () const;
    bool      unbiased       () const;
  };
};

#endif

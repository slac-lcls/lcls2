#include "xtcdata/xtc/L1AcceptEnv.hh"

static const unsigned TrimValue = 1 << 31;
static const unsigned BiasValue = 1 << 30;

XtcData::L1AcceptEnv::L1AcceptEnv()
{
}

XtcData::L1AcceptEnv::L1AcceptEnv(unsigned groups) : Env(groups)
{
}

XtcData::L1AcceptEnv::L1AcceptEnv(unsigned groups, L3TResult l3t)
: Env((groups & ((1 << MaxReadoutGroups) - 1)) | (unsigned(l3t) << MaxReadoutGroups))
{
}

XtcData::L1AcceptEnv::L1AcceptEnv(unsigned groups, L3TResult l3t, bool trim, bool unbiased)
: Env((groups & ((1 << MaxReadoutGroups) - 1)) | (unsigned(l3t) << MaxReadoutGroups) | (trim ? TrimValue : 0) |
      (unbiased ? 0 : BiasValue))
{
}

uint32_t XtcData::L1AcceptEnv::clientGroupMask() const
{
    return value() & ((1 << MaxReadoutGroups) - 1);
}

XtcData::L1AcceptEnv::L3TResult XtcData::L1AcceptEnv::l3t_result() const
{
    return L3TResult((value() >> MaxReadoutGroups) & 3);
}

bool XtcData::L1AcceptEnv::trimmed() const
{
    return value() & TrimValue;
}

bool XtcData::L1AcceptEnv::unbiased() const
{
    return !(value() & BiasValue);
}

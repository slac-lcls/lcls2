#ifndef PDS_ENV
#define PDS_ENV

#include <stdint.h>

namespace XtcData
{
class Env
{
    public:
    Env()
    {
    }
    Env(const Env& in) : _env(in._env)
    {
    }
    Env(uint32_t env);
    uint32_t value() const;

    const Env& operator=(const Env& that);

    private:
    uint32_t _env;
};
}

inline const XtcData::Env& XtcData::Env::operator=(const XtcData::Env& that)
{
    _env = that._env;
    return *this;
}

inline XtcData::Env::Env(uint32_t env) : _env(env)
{
}

inline uint32_t XtcData::Env::value() const
{
    return _env;
}

#endif

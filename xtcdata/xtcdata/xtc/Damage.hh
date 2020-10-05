#ifndef XtcData_Damage_hh
#define XtcData_Damage_hh

#include <stdint.h>

namespace XtcData
{

class Damage
{
public:
    enum Value {
        Truncated           =  0,
        OutOfOrder          =  1,
        OutOfSynch          =  2,
        Corrupted           =  3,
        DroppedContribution =  4,
        MissingData         =  5,
        TimedOut            =  6,
        UserDefined         = 12
    };
    // reserve the top byte to augment user defined errors
    enum { UserBitMask  = 0xf000, UserBitShift = 12 };
    enum { ValueBitMask = 0x0fff };

    Damage()
    {
    }
    Damage(uint16_t v) : _damage(v)
    {
    }
    uint16_t value() const
    {
        return _damage;
    }
    void increase(Damage::Value v)
    {
        _damage |= ((1 << v) & ValueBitMask);
    }
    void increase(uint16_t v)
    {
        _damage |= v & ValueBitMask;
    }
    uint16_t bits() const
    {
        return _damage & ValueBitMask;
    }
    uint16_t userBits() const
    {
        return _damage >> UserBitShift;
    }
    void userBits(uint16_t v)
    {
        _damage &= ValueBitMask;
        _damage |= (v << UserBitShift);
    }

private:
    uint16_t _damage;
};
}

#endif

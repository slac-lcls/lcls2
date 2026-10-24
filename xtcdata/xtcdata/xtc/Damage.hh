/**
 * @file
 * @brief Declares XtcData::Damage, the 16-bit damage word carried by every Xtc.
 */
#ifndef XtcData_Damage_hh
#define XtcData_Damage_hh

#include <stdint.h>

namespace XtcData
{

/** 16-bit damage word. Bits 11:0 (ValueBitMask) are damage flags, one bit per Damage::Value; bits 15:12 (UserBitMask) are user bits. */
class Damage
{
public:
    /** Bit positions of the damage flags; increase(Damage::Value) sets bit (1 << value). */
    enum Value {
        Truncated           =  0,  ///< Bit 0.
        OutOfOrder          =  1,  ///< Bit 1.
        OutOfSynch          =  2,  ///< Bit 2.
        Corrupted           =  3,  ///< Bit 3. XtcIterator::iterate() does not descend into an xtc whose damage has this bit set.
        DroppedContribution =  4,  ///< Bit 4.
        MissingData         =  5,  ///< Bit 5.
        TimedOut            =  6,  ///< Bit 6.
        UserDefined         = 12  ///< Value 12. increase(Damage::Value) masks (1 << 12) with ValueBitMask, so it sets no bit for this value.
    };
    // reserve the top byte to augment user defined errors
    /** Position of the user bits (bits 15:12) in the damage word. */
    enum { UserBitMask  = 0xf000, /**< 0xf000, mask of the user bits. */ UserBitShift = 12  /**< 12, shift of the user bits. */ };
    /** Mask of the damage-flag bits. */
    enum { ValueBitMask = 0x0fff  /**< 0x0fff, bits 11:0. */ };

    /** Default constructor; the damage word is left uninitialized. */
    Damage()
    {
    }
    /** Set the damage word to v. */
    Damage(uint16_t v) : _damage(v)
    {
    }
    /** Return the full 16-bit damage word. */
    uint16_t value() const
    {
        return _damage;
    }
    /** OR bit (1 << v), masked with ValueBitMask, into the damage word. Values of v at or above 12 set nothing. */
    void increase(Damage::Value v)
    {
        _damage |= ((1 << v) & ValueBitMask);
    }
    /** OR (v & ValueBitMask) into the damage word; the user bits are not changed. */
    void increase(uint16_t v)
    {
        _damage |= v & ValueBitMask;
    }
    /** Return the damage word masked with ValueBitMask (bits 11:0). */
    uint16_t bits() const
    {
        return _damage & ValueBitMask;
    }
    /** Return the damage word shifted right by UserBitShift (bits 15:12). */
    uint16_t userBits() const
    {
        return _damage >> UserBitShift;
    }
    /** Clear bits 15:12 and OR in (v << UserBitShift), truncated to 16 bits; bits 11:0 are kept. */
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

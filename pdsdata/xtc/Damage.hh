#ifndef Pds_Damage_hh
#define Pds_Damage_hh

#include <stdint.h>

namespace Pds {

  class Damage {
  public:
    enum Value {
      DroppedContribution    = 1,
      Uninitialized          = 11,
      OutOfOrder             = 12,
      OutOfSynch             = 13,
      UserDefined            = 14,
      IncompleteContribution = 15,
      ContainsIncomplete     = 16
    };
    // reserve the top byte to augment user defined errors
    enum {NotUserBitsMask=0x00FFFFFF, UserBitsShift = 24};

    Damage() {}
    Damage(uint32_t v) : _damage(v) {}
    uint32_t  value() const             { return _damage; }
    void     increase(Damage::Value v)  { _damage |= ((1<<v) & NotUserBitsMask); }
    void     increase(uint32_t v)       { _damage |= v & NotUserBitsMask; }
    uint32_t bits() const               { return _damage & NotUserBitsMask;}
    uint32_t userBits() const           { return _damage >> UserBitsShift; }
    void     userBits(uint32_t v) {
      _damage &= NotUserBitsMask;
      _damage |= (v << UserBitsShift);
    }
    
  private:
    uint32_t _damage;
  };
}

#endif

#ifndef Pds_Eb_SlotTracker_hh
#define Pds_Eb_SlotTracker_hh

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace Pds {
  namespace Eb {

    // Which pulse ID holds each (source, buffer slot) of an RDMA region, so a
    // remote write into a slot still in use is caught when it arrives.
    // Not thread safe: claim() and release() must come from the same thread.
    class SlotTracker
    {
    public:
      static constexpr uint64_t Free = 0;
    public:
      void     resize(unsigned src, size_t nSlots);
      void     clear();
      uint64_t claim(unsigned src, size_t slot, uint64_t pid, uint32_t imm);
      uint64_t release(unsigned src, size_t slot);
      void     dump(const char* who) const;
    public:
      size_t          size(unsigned src)  const { return src < _slots.size() ? _slots[src].size() : 0; }
      static uint64_t pid (uint64_t held) { return held & PidMask; }
    private:
      // Pulse IDs are 56 bits, so a held slot is never Free even for PID 0
      static constexpr uint64_t PidMask = (1ull << 56) - 1;
      static constexpr uint64_t HeldBit = 1ull << 63;
      struct Trace
      {
        uint64_t pid;
        uint64_t held;                  // Slot's prior contents
        size_t   slot;
        uint32_t imm;
        uint16_t src;
        char     op;                    // 'C'laim or 'R'elease
      };
      void _trace(char op, unsigned src, size_t slot, uint64_t pid, uint64_t held, uint32_t imm);
    private:
      std::vector<std::vector<uint64_t>> _slots;
      std::array<Trace, 64>              _ring{};
      unsigned                           _head{0};
    };
  };
};


inline
void Pds::Eb::SlotTracker::resize(unsigned src, size_t nSlots)
{
  if (src >= _slots.size())  _slots.resize(src + 1);
  _slots[src].assign(nSlots, Free);
}

inline
void Pds::Eb::SlotTracker::clear()
{
  for (auto& slots : _slots)  slots.assign(slots.size(), Free);
  _ring.fill({});
  _head = 0;
}

inline
void Pds::Eb::SlotTracker::_trace(char op, unsigned src, size_t slot, uint64_t pid, uint64_t held, uint32_t imm)
{
  _ring[_head] = {pid, held, slot, imm, uint16_t(src), op};
  _head = (_head + 1) % _ring.size();
}

// Returns Free on success, else what holds the slot, which is left unchanged
inline
uint64_t Pds::Eb::SlotTracker::claim(unsigned src, size_t slot, uint64_t pid, uint32_t imm)
{
  auto& held = _slots[src][slot];
  auto  prv  = held;
  _trace('C', src, slot, pid, prv, imm);
  if (prv != Free) [[unlikely]]  return prv;
  held = (pid & PidMask) | HeldBit;
  return Free;
}

// Returns what held the slot, Free if nothing did
inline
uint64_t Pds::Eb::SlotTracker::release(unsigned src, size_t slot)
{
  auto& held = _slots[src][slot];
  auto  prv  = held;
  _trace('R', src, slot, pid(prv), prv, 0);
  held = Free;
  return prv;
}

inline
void Pds::Eb::SlotTracker::dump(const char* who) const
{
  fprintf(stderr, "*** %s slot trace, oldest first:\n", who);
  for (unsigned i = 0; i < _ring.size(); ++i)
  {
    const auto& t = _ring[(_head + i) % _ring.size()];
    if (!t.op)  continue;
    fprintf(stderr, "***   %c src %2u slot %6zu pid %014lx imm %08x, was %s %014lx\n",
            t.op, t.src, t.slot, t.pid, t.imm,
            t.held != Free ? "held by" : "free", pid(t.held));
  }
}

#endif

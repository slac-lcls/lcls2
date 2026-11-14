#pragma once

#include <atomic>
#include <cstdint>

namespace Pds
{

// Written by one thread and read by any, e.g. the metrics exporter
class SharedCounter
{
public:
    SharedCounter() : m_value(0) {}
    void operator++()           { *this += 1; }
    void operator+=(uint64_t n) { m_value.store(m_value.load(std::memory_order_relaxed) + n,
                                                std::memory_order_release); }
    operator uint64_t() const   { return m_value.load(std::memory_order_acquire); }
private:
    std::atomic<uint64_t> m_value;
};

}

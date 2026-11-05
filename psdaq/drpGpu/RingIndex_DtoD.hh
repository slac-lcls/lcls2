#ifndef RINGINDEX_DTOD_HH
#define RINGINDEX_DTOD_HH

#include <cassert>

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/std/atomic>

#include "gpuUtils.hh"                 // hostAlloc

namespace Drp {
  namespace Gpu {

class RingIndexDtoD
{
public:
  __host__ RingIndexDtoD(const unsigned capacity) :
    m_head_d       (nullptr),
    m_tail_d       (nullptr),
    m_capacityMask (capacity-1)     // Range of the buffer index [0, capacity-1]
  {
    assert((capacity & (capacity - 1)) == 0);  // Capacity must be a power of 2

    // The ring is empty when head == tail
    // Head refers to the next index to be allocated
    // Tail refers to the next index after the last freed one
    chkError(cudaMalloc(&m_head_d, 2 * sizeof(*m_head_d))); // Adjacent, so one copy reads both
    m_tail_d = m_head_d + 1;
    reset();
  }

  __host__ ~RingIndexDtoD()
  {
    if (m_head_d)  chkError(cudaFree(m_head_d));
  }

  __device__ bool push(unsigned index) const           /** Move head forward when not full */
  {
    using namespace cuda;
    auto tail = m_tail_d->load(memory_order_acquire);
    auto head = m_head_d->load(memory_order_acquire);
    //if (index != head)  printf("### Expected index %u, got %u\n", head, index);
    auto next = (head+1)  & m_capacityMask;
    if (next == tail)  return false;                   // Full: caller retries to wait for tail to advance
    m_head_d->store(next, memory_order_release);       // Publish new head
    return true;
  }

  __device__ bool pop(unsigned* const __restrict__ index) const /** Return current head */
  {
    using namespace cuda;
    auto tail = m_tail_d->load(memory_order_acquire);
    auto head = m_head_d->load(memory_order_acquire);
    if (tail == head)  return false;                   // Empty: caller retries to wait for head to advance
    *index = tail;                                     // Caller now processes buffer at [tail]
    auto next = (tail+1) & m_capacityMask;
    m_tail_d->store(next, memory_order_release);       // Publish new tail
    return true;
  }

  // On the host these copy from the device, so the caller must have the CUDA context current
  __host__ __device__ unsigned head() const { return _load(m_head_d); }
  __host__ __device__ unsigned tail() const { return _load(m_tail_d); }

  __host__ __device__ unsigned occupancy() const
  {
#ifdef __CUDA_ARCH__
    auto tail = this->tail();
    return (head() - tail) & m_capacityMask;
#else
    unsigned headTail[2]{0, 0};
    chkError(cudaMemcpy(headTail, m_head_d, sizeof(headTail), cudaMemcpyDeviceToHost));
    return (headTail[0] - headTail[1]) & m_capacityMask;
#endif
  }

  __host__ __device__ size_t size() const
  {
    return m_capacityMask + 1;
  }

  __host__ void reset()
  {
    static_assert(sizeof(*m_head_d) == sizeof(unsigned));
    chkError(cudaMemset(m_head_d, 0, 2 * sizeof(*m_head_d)));
  }

private:
  using DevIndex = cuda::atomic<unsigned, cuda::thread_scope_device>;

  __host__ __device__ static unsigned _load(DevIndex const* const index)
  {
#ifdef __CUDA_ARCH__
    return index->load(cuda::memory_order_acquire);
#else
    unsigned value{0};
    chkError(cudaMemcpy(&value, index, sizeof(value), cudaMemcpyDeviceToHost));
    return value;
#endif
  }

private:
  DevIndex*      m_head_d;              // Device memory, so push and pop never cross PCIe
  DevIndex*      m_tail_d;
  const unsigned m_capacityMask;
};

  }
}

#endif // RINGINDEX_DTOD_H

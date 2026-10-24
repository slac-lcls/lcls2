/**
 * @file
 * @brief AlignmentAllocator, a standard-library allocator that returns N-byte aligned storage (adapted from a Stack Overflow answer, per the file comment).
 */
// Modified from:
//   https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
//
// This allocator can be used with stdlib containers that can take an allocator.

#ifndef ALIGNMENT_ALLOCATOR_HH
#define ALIGNMENT_ALLOCATOR_HH

#include <stdlib.h>

namespace Pds
{
  /** Stateless allocator for standard containers that allocates with posix_memalign() at N-byte alignment (default 16). */
  template <typename T, std::size_t N = 16>
  class AlignmentAllocator {
  public:
    typedef T value_type;  ///< Element type T.
    typedef std::size_t size_type;  ///< Size type (std::size_t).
    typedef std::ptrdiff_t difference_type;  ///< Pointer difference type (std::ptrdiff_t).

    typedef T * pointer;  ///< Pointer to T.
    typedef const T * const_pointer;  ///< Pointer to const T.

    typedef T & reference;  ///< Reference to T.
    typedef const T & const_reference;  ///< Reference to const T.

  public:
    /** Does nothing; empty body. */
    inline AlignmentAllocator () throw () { }

    /** Converting constructor from an allocator of another element type; does nothing. */
    template <typename T2>
    inline AlignmentAllocator (const AlignmentAllocator<T2, N> &) throw () { }

    /** Does nothing; empty body. */
    inline ~AlignmentAllocator () throw () { }

    /** Return the address of r. */
    inline pointer address (reference r) {
      return &r;
    }

    /** Return the address of the const element r. */
    inline const_pointer address (const_reference r) const {
      return &r;
    }

    /** Return n elements of storage aligned to N bytes from posix_memalign(), or nullptr if it fails. */
    inline pointer allocate (size_type n) {
      // Revisit: Apparently RHEL6 compiler doesn't have aligned_alloc()
      //return (pointer)aligned_alloc(N, n*sizeof(value_type));

      void* ptr;
      int rc = posix_memalign(&ptr, N, n*sizeof(value_type));
      return (pointer)(rc == 0 ? ptr : nullptr);
    }

    /** Free p with free(); the size argument is not used. */
    inline void deallocate (pointer p, size_type) {
      free (p);
    }

    /** Copy-construct wert in place at p. */
    inline void construct (pointer p, const value_type & wert) {
      new (p) value_type (wert);
    }

    /** Call the destructor of the element at p. */
    inline void destroy (pointer p) {
      p->~value_type ();
    }

    /** Return the largest size_type value divided by the element size. */
    inline size_type max_size () const throw () {
      return size_type (-1) / sizeof (value_type);
    }

    /** Gives the same allocator for element type T2 (as other). */
    template <typename T2>
    struct rebind {
      typedef AlignmentAllocator<T2, N> other;  ///< The allocator for element type T2 with the same alignment N.
    };

    /** Return the negation of operator==, which is always true, so this always returns false. */
    bool operator!=(const AlignmentAllocator<T,N>& other) const  {
      return !(*this == other);
    }

    // Returns true if and only if storage allocated from *this
    // can be deallocated from other, and vice versa.
    // Always returns true for stateless allocators.
    /** Always returns true, since the allocator holds no state (per the code comment). */
    bool operator==(const AlignmentAllocator<T,N>& other) const {
      return true;
    }
  };
};

#endif

/**
 * @file
 * @brief Declares the Allocator interface and two implementations: Stack (bump allocator over a fixed buffer) and Heap (malloc/free).
 */
#ifndef ALLOCATOR__H
#define ALLOCATOR__H

#include <stdint.h>
#include <stdlib.h>

/** Abstract allocator interface with malloc() and free(). */
class Allocator{ //Heap -> Allocator
public:
    /** Allocate size bytes. Pure virtual. */
    virtual void *malloc(size_t size) = 0;
    /** Release ptr. Pure virtual. */
    virtual void free(void *ptr) = 0;
};

/**
 * Bump allocator over a 1 MiB member buffer: malloc() returns the current position and advances it by size with no bounds check, and free() does nothing.
 * Both overrides are private, so they are reachable only through an Allocator pointer or reference.
 */
class Stack:public Allocator{ // PebbleHeap -> Stack
public:
    /** Start allocating at the beginning of the internal buffer. */
    Stack():_allocator(_buf){}

private:
    uint8_t _buf[1024*1024];
    uint8_t *_allocator;

    virtual void free(void *ptr) {
    }

    virtual void *malloc(size_t size) {
        void *curr_allocator = _allocator;
        _allocator += size;
        return curr_allocator;
    }
};

/** Allocator whose malloc() and free() call ::malloc and ::free. Both overrides are private, so they are reachable only through an Allocator pointer or reference. */
class Heap:public Allocator{ // StandardHeap -> Heap
public:
    /** Default constructor; does nothing. */
    Heap(){}

private:
    virtual void free(void *ptr) {
        return ::free(ptr);
    }
    virtual void *malloc(size_t size) {
        void *ptr = ::malloc(size);
        return ptr;
    }
};

#endif // ALLOCATOR__H

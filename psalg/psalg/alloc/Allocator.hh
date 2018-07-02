#ifndef ALLOCATOR__H
#define ALLOCATOR__H

#include <stdint.h>
#include <stdlib.h>

class Allocator{ //Heap -> Allocator
public:
    virtual void *malloc(size_t size) = 0;
    virtual void free(void *ptr) = 0;
};

class Stack:public Allocator{ // PebbleHeap -> Stack
public:
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

class Heap:public Allocator{ // StandardHeap -> Heap
public:
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

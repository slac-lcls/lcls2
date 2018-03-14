#ifndef HEAP__H
#define HEAP__H

#include <iostream>
#include "xtcdata/xtc/ShapesData.hh"

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
        //std::cout << "**** Stack free (No Op): " << ptr << std::endl;
    }

    virtual void *malloc(size_t size) {
        void *curr_allocator = _allocator;
        _allocator += size;
        //std::cout << "**** Stack malloc: " << curr_allocator << std::endl;
        return curr_allocator;
    }
};

class Heap:public Allocator{ // StandardHeap -> Heap
public:
    Heap(){}

private:
    virtual void free(void *ptr) {
        //std::cout << "**** Heap free: " << ptr << std::endl;
        return ::free(ptr);
    }
    virtual void *malloc(size_t size) {
        void *ptr = ::malloc(size);
        //std::cout << "**** Heap malloc: " << ptr << std::endl;
        return ptr;
    }
};

#endif // HEAP__H
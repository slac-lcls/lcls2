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
        std::cout << "**** Stack free" << std::endl;
    }

    virtual void *malloc(size_t size) {
        std::cout << "**** Stack malloc" << std::endl;
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
        std::cout << "**** Heap free" << std::endl;
        return ::free(ptr);
    }
    virtual void *malloc(size_t size) {
        std::cout << "**** Heap malloc" << std::endl;
        return ::malloc(size);
    }
};

/*
class Pebble
{
public:
    void* fex_data() {return reinterpret_cast<void*>(_fex_buffer);}
    PGPData* pgp_data;

private:
    uint8_t _fex_buffer[1024*1024];
    Allocator _allocator;
};
*/

#endif // HEAP__H
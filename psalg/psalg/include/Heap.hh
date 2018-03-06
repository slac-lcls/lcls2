#ifndef HEAP__H
#define HEAP__H

#include <iostream>
#include "xtcdata/xtc/ShapesData.hh"

class Heap{
public:
    virtual void *malloc_array(size_t size){
        return malloc(size+sizeof(XtcData::Shape));
    }
    virtual void free(void *ptr) = 0;
protected:
    virtual void *malloc(size_t size) = 0;

};

class PebbleHeap:public Heap{
public:
    PebbleHeap():_heap(_heap_buffer){}
    void free(void *ptr){}
private:
    uint8_t _heap_buffer[1024*1024];
    uint8_t *_heap;

    friend class Heap;
    void *malloc(size_t size) {
        std::cout << "pebble malloc" << std::endl;
        void *curr_heap = _heap;
        _heap += size;
        return curr_heap;
    }
};

class StandardHeap:public Heap{
public:
    StandardHeap(){}
    void free(void *ptr){
        return ::free(ptr);
    }
private:
    friend class Heap;
    void *malloc(size_t size) {
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
    Heap _heap;
};
*/

#endif // HEAP__H
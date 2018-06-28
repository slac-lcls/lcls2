//g++ -g -Wall -std=c++11 -I /reg/neh/home/yoon82/temp/lcls2/install/include test_array.cpp -o test_array
//valgrind ./test_array

#include <iostream>
#include <vector>
#include <stdlib.h>

#include "psalg/AllocArray.hh"
#include "psalg/Allocator.hh"

using namespace psalg;

class Foo{
public:
    Foo(Allocator& allocator, const size_t& seg):_a(allocator, 3),
                                                 _b(allocator, 4),
                                                 _allocator(allocator),
                                                 m_seg(seg)
    {
    }

    void set(){
        _a.push_back(91);
        _a.push_back(92);
        _a.push_back(93);
    }

    float get(int i){
        return _a(i);
    }

    size_t seg(){
        return m_seg;
    }

private:
    AllocArray1D<float> _a;
    AllocArray1D<AllocArray1D<float> > _b;
    Allocator& _allocator;
    size_t m_seg;
};

void testArray(Allocator& buf1){
    // Initialize shape and memory
    uint32_t *shape = new uint32_t[5];
    shape[0] = 2;
    shape[1] = 3;
    shape[2] = 4;
    shape[3] = 5;
    shape[4] = 6;
    uint8_t *buf = new uint8_t[1024*1024];

    std::cout << "----- Array" << std::endl;
    unsigned rank = 1;
    auto a = Array<float>(buf, shape, rank);
    assert(a.rank()==rank);
    assert(a.shape()[0]==shape[0] && a.shape()[4]==shape[4]);
    a(0)=999;
    a(1)=998;
    assert(a(1)==998);

    std::cout << "----- NDArray" << std::endl;

    auto m = AllocArray<float>(buf1, 6, 2);
    m.shape(2,3);
    for (unsigned i = 0; i < m.shape()[0]; i++) {
    for (unsigned j = 0; j < m.shape()[1]; j++) {
        m(i,j) = 3*i+j;
    }
    }
    assert(m(0,0)==0 && m(0,1)==1);
    assert(m(1,0)==3 && m(1,2)==5);
    assert(m.refCnt()==1);
    assert(m.num_elem()==6);

    std::cout << "####### Test AllocArray1D push_back" << std::endl;

    auto b = AllocArray1D<float>(buf1, 3);
    b.push_back(99);
    b.push_back(98);
    b.push_back(97);
    assert(b.refCnt()==1);
    assert(b(0)==99 && b(2)==97);
    assert(b.shape()[0]==3 && b.shape()[1]==0);

    std::cout << "####### Test copy" << std::endl;

    auto c(b);
    assert(c.refCnt()==2 && c.refCnt()==b.refCnt()); // b and c share the same refCnt
    assert(c(1)==98);
    b(1)=93;
    assert(b(1)==c(1)); // b and c share the same data

    std::cout << "####### Test assignment Case 1" << std::endl;
    c=b;
    assert(c.refCnt()==2 && c.refCnt()==b.refCnt());
    assert(c.shape()[0]==3 && c.shape()[1]==0);

    std::cout << "####### Test assignment Case 2" << std::endl;
    auto d = AllocArray1D<float>(buf1, 2);
    d=b;
    assert(d.refCnt()==3 && d.refCnt()==b.refCnt());
    assert(d(0)==99 && d(1)==93);
    assert(d.shape()[0]==3 && d.shape()[1]==0);

    std::cout << "####### Test assignment Case 3" << std::endl;
    b=b;
    assert(b.refCnt()==3);

    std::cout << "####### Array of Arrays" << std::endl;
    auto f = AllocArray1D<AllocArray1D<float> >(buf1, 3);
    assert(f.refCnt()==1);
    assert(f.size()==0 && f.capacity()==3);
    f.push_back(b);
    assert(b.refCnt()==4);
    assert(f.num_elem()==1);

    auto g = AllocArray1D<float>(buf1, 100000);
    for (int i =0; i<100000; i++) g.push_back(66+i);
    f.push_back(g);
    assert(f.shape()[0]==2 && f.num_elem()==2);

    auto h = AllocArray1D<float>(buf1, 4);
    for (int i =0; i<4; i++) h.push_back(77+i);
    f.push_back(h);
    assert(f.shape()[0]==3);
    assert(f.refCnt()==1);

    std::cout << "####### Test Array of Arrays clear 1" << std::endl;
    f.clear();
    f.push_back(h);
    f.push_back(g);
    f.push_back(b);

    assert(f(0)(1)==78);
    assert(f(1)(10)==76);
    assert(f(2)(0)==99);
    assert(b.refCnt()==4);

    std::cout << "####### Test Array of Arrays clear 2" << std::endl;
    b.clear();
    assert(b.size()==0 && b.capacity()==3);
    assert(b.refCnt()==4);

    std::cout << "----- Class using Array" << std::endl;
    Foo foo(buf1, 9);
    foo.set();
    assert(foo.get(1)==92);
    assert(foo.seg()==9);

    #ifndef NDEBUG
      std::cout << "DONE in Debug mode!" << std::endl; // DEBUG
    #else
      std::cout << "DONE in Release mode!" << std::endl;  // RELEASE
    #endif



    delete[] shape;
    delete[] buf;
}

int main () {

    // Test with heap
    Heap buf1;
    testArray(buf1);

    // Test with stack
    Stack buf2;
    testArray(buf2);

    return 0;
}

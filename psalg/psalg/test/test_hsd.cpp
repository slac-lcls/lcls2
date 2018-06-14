//g++ -g -Wall -std=c++11 -I /reg/neh/home/yoon82/Software/lcls2/install/include test_hsd.cpp -o test_hsd
//valgrind ./test_hsd

#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <fstream>

#include "psalg/include/AllocArray.hh"
#include "psalg/include/Allocator.hh"
#include "psalg/include/hsd.hh"

//using namespace psalgos;
using namespace psalg;

int main () {
    //open file
    std::ifstream infile("/reg/neh/home/yoon82/Software/lcls2/chan0_e0.bin");
    //get length of file
    infile.seekg(0, infile.end);
    size_t length = infile.tellg();
    infile.seekg(0, infile.beg);
    char buffer[length];
    //read file
    infile.read(buffer, length);

    // Test with heap
    Heap heap;
    unsigned nChan = 4;

    Pds::HSD::Client *pClient = new Pds::HSD::Client(&heap, "1.0.0", nChan);
    Pds::HSD::HsdEventHeaderV1 *pHsd = pClient->getHsd();
    Pds::HSD::Hsd_v1_0_0 *vHsd = (Pds::HSD::Hsd_v1_0_0*) pHsd;
    for (unsigned i = 0; i < nChan; i++) {
        vHsd->parseChan((const uint8_t*)buffer, i);
    }

    delete pClient;

    return 0;
}


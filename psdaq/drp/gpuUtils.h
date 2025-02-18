/**
 * ----------------------------------------------------------------------------
 * Company    : SLAC National Accelerator Laboratory
 * ----------------------------------------------------------------------------
 * Description: General utilities for C/C++/CUDA code
 * ----------------------------------------------------------------------------
 * This file is part of 'axi-pcie-devel'. It is subject to
 * the license terms in the LICENSE.txt file found in the top-level directory
 * of this distribution and at:
 *    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
 * No part of 'axi-pcie-devel', including this file, may be copied, modified,
 * propagated, or distributed except according to the terms contained in
 * the LICENSE.txt file.
 * ----------------------------------------------------------------------------
 **/
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

inline void putByte(uint8_t byte) {
    putc("0123456789ABCDEF"[byte & 0xF], stdout);
    putc("0123456789ABCDEF"[(byte>>8) & 0xF], stdout);
}

/**
 * Dump memory to stdout, in a (kinda) pretty format
 * However, the code is far from pretty.
 */
template<typename T, int COLS = 16>
static inline void dumpMem(T* ptr, size_t sizeInBytes)
{
    assert(sizeInBytes % sizeof(T) == 0);
    sizeInBytes /= sizeof(T);
    printf("0x%-5lX ", 1L);
    for (int i = 0; i < sizeInBytes; ++i, ++ptr) {
        const char* p = (const char*)ptr;
        for (int b = 0; b < sizeof(T); ++b, ++p)
            putByte(*p);

        if ((i+1) % COLS == 0 && i > 0) {
            putc('\n', stdout);
            if (i+1 < sizeInBytes)
                printf("0x%-5lX ", (i+1)*sizeof(T));
        }
        else
            putc(' ', stdout);
    }
}

/**
 * Stupid wrapper for writing to FPGA registers
 * \param fpgaRegs Pointer to DMA space
 * \param reg Register offset
 */
template<typename T>
static inline void writeRegister(void* fpgaRegs, uintptr_t reg, const T& value)
{
	*(T*)(((uint8_t*)fpgaRegs) + reg) = value;
}

/**
 * Stupid wrapper for reading FPGA registers mapped via DMA
 */
template<typename T>
static inline T readRegister(void* fpgaRegs, uintptr_t reg)
{
	return *(T*)(((uint8_t*)fpgaRegs) + reg);
}

static inline double curTime() {
    timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return double(tp.tv_sec) + (double(tp.tv_nsec) / 1e9);
}

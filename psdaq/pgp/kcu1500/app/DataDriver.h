/**
 *-----------------------------------------------------------------------------
 * Title      : Data Development Card Driver, Shared Header
 * ----------------------------------------------------------------------------
 * File       : DataDriver.h
 * Created    : 2017-03-21
 * ----------------------------------------------------------------------------
 * Description:
 * Defintions and inline functions for interacting with Data Development driver.
 * ----------------------------------------------------------------------------
 * This file is part of the aes_stream_drivers package. It is subject to 
 * the license terms in the LICENSE.txt file found in the top-level directory 
 * of this distribution and at: 
 *    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
 * No part of the aes_stream_drivers package, including this file, may be 
 * copied, modified, propagated, or distributed except according to the terms 
 * contained in the LICENSE.txt file.
 * ----------------------------------------------------------------------------
**/
#ifndef __DATA_DRIVER_H__
#define __DATA_DRIVER_H__
#include "AxisDriver.h"
#include "DmaDriver.h"
#include "FpgaProm.h"
#include "AxiVersion.h"

inline uint32_t dmaDest(uint32_t lane, uint32_t vc) 
{
  return lane*32 + vc;
}
#endif


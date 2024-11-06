
#pragma once

/*****************************************************************
 * AxiGpuAsyncCore
 *****************************************************************/

/** Common registers **/

#define GPU_ASYNC_INFO1_REG					0x4
#define GPU_ASYNC_INFO1_ARCACHE(_val)		((_val) & 0xFF)
#define GPU_ASYNC_INFO1_AWCACHE(_val)		(((_val) >> 8) & 0xFF)
#define GPU_ASYNC_INFO1_BYTES(_val)			(((_val) >> 16) & 0xFF)
#define GPU_ASYNC_INFO1_MAX_BUFFERS(_val)	(((_val) >> 24) & 0x1F)	/* 5 bits */

#define GPU_ASYNC_INFO2_REG					0x8
#define GPU_ASYNC_INFO2_WR_CNT(_val)		((_val) & 0xFF)
#define GPU_ASYNC_INFO2_WR_EN(_val)			(((_val) >> 8) & 0xFF)
#define GPU_ASYNC_INFO2_RD_CNT(_val)		(((_val) >> 16) & 0xFF)
#define GPU_ASYNC_INFO2_RD_EN(_val)			(((_val) >> 24) & 0xFF)

#define GPU_ASYNC_RX_FRAME_CNT				0x10
#define GPU_ASYNC_TX_FRAME_CNT				0x14
#define GPU_ASYNC_WR_ERR_CNT				0x18
#define GPU_ASYNC_RD_ERR_CNT				0x1C

#define GPU_ASYNC_CNT_RST					0x20

/** Write addresses and sizes [Starts at 0x100] **/
#define GPU_ASYNC_WR_ADDR(_n)				(0x100 | ((_n) << 6))
#define GPU_ASYNC_WR_SIZE(_n)				(0x108 | ((_n) << 6))

/** Read addresses [Starts at 0x200] **/
#define GPU_ASYNC_RD_ADDR(_n)				(0x200 | ((_n) << 6))

/** Write enable bit [Starts at 0x300] **/
#define GPU_ASYNC_WR_ENABLE(_n) 			(0x300 | ((_n) * 4))

/** Read enable bit [Starts at 0x400] **/
#define GPU_ASYNC_RD_ENABLE(_n) 			(0x400 | ((_n) * 4))
#define GPU_ASYNC_RD_SIZE(_n)	 			(0x400 | ((_n) * 4))

/** I/O Stats [Starts at 0x500] **/
#define GPU_ASYNC_TOTAL_LATENCY(_n) 		(0x500 + (_n*16))
#define GPU_ASYNC_GPU_LATENCY(_n) 			(0x504 + (_n*16))
#define GPU_ASYNC_WR_LATENCY(_n)			(0x508 + (_n*16))
#define GPU_ASYNC_RD_LATENCY(_n)			(0x50C + (_n*16))

/** Max buffers, must match firmware value **/
#define MAX_BUFFERS                         4


/*****************************************************************
 * AxiPcieCore
 *****************************************************************/

#define PCIE_AXI_VERSION_OFFSET             0x20000

/*****************************************************************
 * PcieAxiVersion
 *****************************************************************/

#define PCIE_AXI_VERSION_SCRATCHPAD         0x4
#define PCIE_AXI_VERSION_CLK_FREQ           (0x400+(4*8))

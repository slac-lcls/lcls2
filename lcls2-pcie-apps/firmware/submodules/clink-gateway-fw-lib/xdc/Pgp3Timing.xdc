##############################################################################
## This file is part of 'Camera link gateway'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'Camera link gateway', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

create_generated_clock -name pgpTxClk   [get_pins {U_Core/GEN_PGP3.U_PGP/U_PGPv3/REAL_PGP.U_TX_PLL/PllGen.U_Pll/CLKOUT0}] 
create_generated_clock -name pgpRxClk00 [get_pins {U_Core/GEN_PGP3.U_PGP/U_PGPv3/REAL_PGP.GEN_LANE[0].U_Pgp/U_Pgp3Gtx7IpWrapper/U_RX_PLL/PllGen.U_Pll/CLKOUT0}] 
create_generated_clock -name pgpRxClk01 [get_pins {U_Core/GEN_PGP3.U_PGP/U_PGPv3/REAL_PGP.GEN_LANE[0].U_Pgp/U_Pgp3Gtx7IpWrapper/U_RX_PLL/PllGen.U_Pll/CLKOUT1}] 
create_generated_clock -name pgpRxClk10 [get_pins {U_Core/GEN_PGP3.U_PGP/U_PGPv3/REAL_PGP.GEN_LANE[1].U_Pgp/U_Pgp3Gtx7IpWrapper/U_RX_PLL/PllGen.U_Pll/CLKOUT0}] 
create_generated_clock -name pgpRxClk11 [get_pins {U_Core/GEN_PGP3.U_PGP/U_PGPv3/REAL_PGP.GEN_LANE[1].U_Pgp/U_Pgp3Gtx7IpWrapper/U_RX_PLL/PllGen.U_Pll/CLKOUT1}] 

set_clock_groups -asynchronous \
    -group [get_clocks -include_generated_clocks -of_objects [get_pins -hier -filter {name=~*gt0_Pgp3Gtx7Ip10G_i*gtxe2_i*TXOUTCLK}]] \
    -group [get_clocks -include_generated_clocks -of_objects [get_pins -hier -filter {name=~*gt0_Pgp3Gtx7Ip10G_i*gtxe2_i*RXOUTCLK}]]

set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins U_Core/GEN_PGP3.U_PGP/U_PGPv3/REAL_PGP.U_TX_PLL/PllGen.U_Pll/CLKOUT1]] -group [get_clocks -of_objects [get_pins U_Core/GEN_PGP3.U_PGP/U_MMCM/MmcmGen.U_Mmcm/CLKOUT1]]

set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins U_Core/GEN_PGP3.U_PGP/U_MMCM/MmcmGen.U_Mmcm/CLKOUT1]] -group [get_clocks -of_objects [get_pins {U_Core/GEN_PGP3.U_PGP/U_PGPv3/REAL_PGP.GEN_LANE[0].U_Pgp/U_Pgp3Gtx7IpWrapper/U_RX_PLL/PllGen.U_Pll/CLKOUT1}]]
set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins U_Core/GEN_PGP3.U_PGP/U_MMCM/MmcmGen.U_Mmcm/CLKOUT1]] -group [get_clocks -of_objects [get_pins {U_Core/GEN_PGP3.U_PGP/U_PGPv3/REAL_PGP.GEN_LANE[1].U_Pgp/U_Pgp3Gtx7IpWrapper/U_RX_PLL/PllGen.U_Pll/CLKOUT1}]]

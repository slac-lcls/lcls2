##############################################################################
## This file is part of 'Camera link gateway'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'Camera link gateway', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

create_generated_clock -name clk156 [get_pins {U_axilClk/PllGen.U_Pll/CLKOUT0}] 
create_generated_clock -name clk25  [get_pins {U_axilClk/PllGen.U_Pll/CLKOUT1}] 

create_generated_clock -name clk238 [get_pins {U_Hardware/U_TimingRx/U_238MHz/MmcmGen.U_Mmcm/CLKOUT0}] 
create_generated_clock -name clk371 [get_pins {U_Hardware/U_TimingRx/U_371MHz/MmcmGen.U_Mmcm/CLKOUT0}] 

create_generated_clock -name clk119 [get_pins {U_Hardware/U_TimingRx/GEN_VEC[0].U_refClkDiv2/O}] 
create_generated_clock -name clk186 [get_pins {U_Hardware/U_TimingRx/GEN_VEC[1].U_refClkDiv2/O}] 

create_clock -name timingRxClk -period 5.384 [get_pins {U_Hardware/U_TimingRx/U_RXCLK/O}] 
create_clock -name timingTxClk -period 5.384 [get_pins {U_Hardware/U_TimingRx/U_TXCLK/O}] 

create_clock -name timingRxClk0    -period 5.384 [get_pins {U_Hardware/U_TimingRx/GEN_VEC[0].REAL_PCIE.U_GTH/LOCREF_G.U_TimingGthCore/inst/gen_gtwizard_gthe3_top.TimingGth_fixedlat_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[0].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/RXOUTCLK}]
create_clock -name timingTxOutClk0 -period 5.384 [get_pins {U_Hardware/U_TimingRx/GEN_VEC[0].REAL_PCIE.U_GTH/LOCREF_G.U_TimingGthCore/inst/gen_gtwizard_gthe3_top.TimingGth_fixedlat_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[0].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/TXOUTCLK}]
create_generated_clock -name timingTxClk0        [get_pins {U_Hardware/U_TimingRx/GEN_VEC[0].REAL_PCIE.U_GTH/LOCREF_G.TIMING_TXCLK_BUFG_GT/O}]

create_clock -name timingRxClk1    -period 5.384 [get_pins {U_Hardware/U_TimingRx/GEN_VEC[1].REAL_PCIE.U_GTH/LOCREF_G.U_TimingGthCore/inst/gen_gtwizard_gthe3_top.TimingGth_fixedlat_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[0].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/RXOUTCLK}]
create_clock -name timingTxOutClk1 -period 5.384 [get_pins {U_Hardware/U_TimingRx/GEN_VEC[1].REAL_PCIE.U_GTH/LOCREF_G.U_TimingGthCore/inst/gen_gtwizard_gthe3_top.TimingGth_fixedlat_gtwizard_gthe3_inst/gen_gtwizard_gthe3.gen_channel_container[0].gen_enabled_channel.gthe3_channel_wrapper_inst/channel_inst/gthe3_channel_gen.gen_gthe3_channel_inst[0].GTHE3_CHANNEL_PRIM_INST/TXOUTCLK}]
create_generated_clock -name timingTxClk1        [get_pins {U_Hardware/U_TimingRx/GEN_VEC[1].REAL_PCIE.U_GTH/LOCREF_G.TIMING_TXCLK_BUFG_GT/O}]

set_case_analysis 1 [get_pins {U_Hardware/U_TimingRx/GEN_VEC[0].REAL_PCIE.U_RXCLK/S}]
set_case_analysis 1 [get_pins {U_Hardware/U_TimingRx/GEN_VEC[1].REAL_PCIE.U_RXCLK/S}]
set_case_analysis 1 [get_pins {U_Hardware/U_TimingRx/U_RXCLK/S}]
set_case_analysis 1 [get_pins {U_Hardware/U_TimingRx/U_TXCLK/S}]

set_clock_groups -asynchronous -group [get_clocks {clk156}] -group [get_clocks {timingRxClk0}]  -group [get_clocks {timingTxClk0}] -group [get_clocks {timingRxClk}] -group [get_clocks {timingTxClk}]
set_clock_groups -asynchronous -group [get_clocks {clk156}] -group [get_clocks {timingRxClk1}]  -group [get_clocks {timingTxClk1}] -group [get_clocks {timingRxClk}] -group [get_clocks {timingTxClk}]

set_clock_groups -asynchronous -group [get_clocks {clk156}]  -group [get_clocks {clk238}]  -group [get_clocks {clk371}] -group [get_clocks {dmaClk}]

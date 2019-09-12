##############################################################################
## This file is part of 'Camera link gateway'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'Camera link gateway', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

# .bit File Configuration
set_property BITSTREAM.CONFIG.CONFIGRATE 50    [current_design] 
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 1   [current_design]
set_property BITSTREAM.CONFIG.SPI_FALL_EDGE No [current_design]

# Clink Ports
set_property DIFF_TERM   TRUE     [get_ports {cbl0Half0P[0]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half0P[0]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half0P[1]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half0P[1]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half0P[2]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half0P[2]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half0P[3]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half0P[3]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half0P[4]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half0P[4]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half1P[0]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half1P[0]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half1P[1]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half1P[1]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half1P[2]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half1P[2]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half1P[3]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half1P[3]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl0Half1P[4]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl0Half1P[4]}]

set_property IOSTANDARD  LVDS_25  [get_ports {cbl0SerP}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half0P[0]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half0P[0]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half0P[1]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half0P[1]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half0P[2]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half0P[2]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half0P[3]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half0P[3]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half0P[4]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half0P[4]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half1P[0]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half1P[0]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half1P[1]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half1P[1]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half1P[2]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half1P[2]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half1P[3]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half1P[3]}]

set_property DIFF_TERM   TRUE     [get_ports {cbl1Half1P[4]}]
set_property IOSTANDARD  LVDS_25  [get_ports {cbl1Half1P[4]}]

set_property IOSTANDARD  LVDS_25  [get_ports {cbl1SerP}]

set_property PACKAGE_PIN Y18      [get_ports {cbl0Half0P[0]}]
set_property PACKAGE_PIN Y19      [get_ports {cbl0Half0M[0]}]

set_property PACKAGE_PIN AA14     [get_ports {cbl0Half0P[1]}]
set_property PACKAGE_PIN AA15     [get_ports {cbl0Half0M[1]}]

set_property PACKAGE_PIN R16      [get_ports {cbl0Half0P[2]}]
set_property PACKAGE_PIN T16      [get_ports {cbl0Half0M[2]}]

set_property PACKAGE_PIN W14      [get_ports {cbl0Half0P[3]}]
set_property PACKAGE_PIN Y14      [get_ports {cbl0Half0M[3]}]

set_property PACKAGE_PIN T15      [get_ports {cbl0Half0P[4]}]
set_property PACKAGE_PIN U15      [get_ports {cbl0Half0M[4]}]

set_property PACKAGE_PIN V19      [get_ports {cbl0Half1P[0]}]
set_property PACKAGE_PIN W19      [get_ports {cbl0Half1M[0]}]

set_property PACKAGE_PIN T21      [get_ports {cbl0Half1P[1]}]
set_property PACKAGE_PIN U21      [get_ports {cbl0Half1M[1]}]

set_property PACKAGE_PIN T18      [get_ports {cbl0Half1P[2]}]
set_property PACKAGE_PIN U18      [get_ports {cbl0Half1M[2]}]

set_property PACKAGE_PIN U17      [get_ports {cbl0Half1P[3]}]
set_property PACKAGE_PIN V18      [get_ports {cbl0Half1M[3]}]

set_property PACKAGE_PIN Y21      [get_ports {cbl0Half1P[4]}]
set_property PACKAGE_PIN Y22      [get_ports {cbl0Half1M[4]}]

set_property PACKAGE_PIN AB15     [get_ports {cbl0SerP}]
set_property PACKAGE_PIN AB16     [get_ports {cbl0SerM}]

set_property PACKAGE_PIN E17      [get_ports {cbl1Half0P[0]}]
set_property PACKAGE_PIN E18      [get_ports {cbl1Half0M[0]}]

set_property PACKAGE_PIN D19      [get_ports {cbl1Half0P[1]}]
set_property PACKAGE_PIN D20      [get_ports {cbl1Half0M[1]}]

set_property PACKAGE_PIN B18      [get_ports {cbl1Half0P[2]}]
set_property PACKAGE_PIN A19      [get_ports {cbl1Half0M[2]}]

set_property PACKAGE_PIN A20      [get_ports {cbl1Half0P[3]}]
set_property PACKAGE_PIN A21      [get_ports {cbl1Half0M[3]}]

set_property PACKAGE_PIN B20      [get_ports {cbl1Half0P[4]}]
set_property PACKAGE_PIN B21      [get_ports {cbl1Half0M[4]}]

set_property PACKAGE_PIN C17      [get_ports {cbl1Half1P[0]}]
set_property PACKAGE_PIN C18      [get_ports {cbl1Half1M[0]}]

set_property PACKAGE_PIN G15      [get_ports {cbl1Half1P[1]}]
set_property PACKAGE_PIN G16      [get_ports {cbl1Half1M[1]}]

set_property PACKAGE_PIN F15      [get_ports {cbl1Half1P[2]}]
set_property PACKAGE_PIN F16      [get_ports {cbl1Half1M[2]}]

set_property PACKAGE_PIN C13      [get_ports {cbl1Half1P[3]}]
set_property PACKAGE_PIN B13      [get_ports {cbl1Half1M[3]}]

set_property PACKAGE_PIN C14      [get_ports {cbl1Half1P[4]}]
set_property PACKAGE_PIN C15      [get_ports {cbl1Half1M[4]}]

set_property PACKAGE_PIN J16      [get_ports {cbl1SerP}]
set_property PACKAGE_PIN J17      [get_ports {cbl1SerM}]

# LEDs
set_property -dict { PACKAGE_PIN G20 IOSTANDARD LVCMOS33 } [get_ports { ledRed[0] }];
set_property -dict { PACKAGE_PIN L18 IOSTANDARD LVCMOS33 } [get_ports { ledGrn[0] }];
set_property -dict { PACKAGE_PIN F20 IOSTANDARD LVCMOS33 } [get_ports { ledBlu[0] }];

set_property -dict { PACKAGE_PIN E21 IOSTANDARD LVCMOS33 } [get_ports { ledRed[1] }];
set_property -dict { PACKAGE_PIN H22 IOSTANDARD LVCMOS33 } [get_ports { ledGrn[1] }];
set_property -dict { PACKAGE_PIN E22 IOSTANDARD LVCMOS33 } [get_ports { ledBlu[1] }];

# Boot Memory Ports
set_property -dict { PACKAGE_PIN L16 IOSTANDARD LVCMOS33 } [get_ports { bootCsL }];
set_property -dict { PACKAGE_PIN H18 IOSTANDARD LVCMOS33 } [get_ports { bootMosi }];
set_property -dict { PACKAGE_PIN H19 IOSTANDARD LVCMOS33 } [get_ports { bootMiso }];

# Timing GPIO Ports
set_property -dict { PACKAGE_PIN C8  IOSTANDARD LVCMOS25 } [get_ports { timingClkSel }];
set_property -dict { PACKAGE_PIN B8  IOSTANDARD LVCMOS25 } [get_ports { timingXbarSel[0] }];
set_property -dict { PACKAGE_PIN A11 IOSTANDARD LVCMOS25 } [get_ports { timingXbarSel[1] }];
set_property -dict { PACKAGE_PIN A10 IOSTANDARD LVCMOS25 } [get_ports { timingXbarSel[2] }];
set_property -dict { PACKAGE_PIN E8  IOSTANDARD LVCMOS25 } [get_ports { timingXbarSel[3] }];

# GTX Ports
set_property PACKAGE_PIN F2 [get_ports { gtTxP[0] }];
set_property PACKAGE_PIN F1 [get_ports { gtTxN[0] }];
set_property PACKAGE_PIN G4 [get_ports { gtRxP[0] }];
set_property PACKAGE_PIN G3 [get_ports { gtRxN[0] }];

set_property PACKAGE_PIN D2 [get_ports { gtTxP[1] }];
set_property PACKAGE_PIN D1 [get_ports { gtTxN[1] }];
set_property PACKAGE_PIN E4 [get_ports { gtRxP[1] }];
set_property PACKAGE_PIN E3 [get_ports { gtRxN[1] }];

set_property PACKAGE_PIN B2 [get_ports { gtTxP[2] }];
set_property PACKAGE_PIN B1 [get_ports { gtTxN[2] }];
set_property PACKAGE_PIN C4 [get_ports { gtRxP[2] }];
set_property PACKAGE_PIN C3 [get_ports { gtRxN[2] }];

set_property PACKAGE_PIN A4 [get_ports { gtTxP[3] }];
set_property PACKAGE_PIN A3 [get_ports { gtTxN[3] }];
set_property PACKAGE_PIN B6 [get_ports { gtRxP[3] }];
set_property PACKAGE_PIN B5 [get_ports { gtRxN[3] }];

set_property PACKAGE_PIN D6 [get_ports { gtClkP[0] }];
set_property PACKAGE_PIN D5 [get_ports { gtClkN[0] }];

set_property PACKAGE_PIN F6 [get_ports { gtClkP[1] }];
set_property PACKAGE_PIN F5 [get_ports { gtClkN[1] }];

# SFP Ports
set_property -dict { PACKAGE_PIN H20 IOSTANDARD LVCMOS33 } [get_ports { sfpScl[0] }];
set_property -dict { PACKAGE_PIN N19 IOSTANDARD LVCMOS33 } [get_ports { sfpSda[0] }];

set_property -dict { PACKAGE_PIN L19 IOSTANDARD LVCMOS33 } [get_ports { sfpScl[1] }];
set_property -dict { PACKAGE_PIN M17 IOSTANDARD LVCMOS33 } [get_ports { sfpSda[1] }];

set_property -dict { PACKAGE_PIN L20 IOSTANDARD LVCMOS33 } [get_ports { sfpScl[2] }];
set_property -dict { PACKAGE_PIN M18 IOSTANDARD LVCMOS33 } [get_ports { sfpSda[2] }];

set_property -dict { PACKAGE_PIN N18 IOSTANDARD LVCMOS33 } [get_ports { sfpScl[3] }];
set_property -dict { PACKAGE_PIN N22 IOSTANDARD LVCMOS33 } [get_ports { sfpSda[3] }];

# Misc Ports
set_property -dict { PACKAGE_PIN F21 IOSTANDARD LVCMOS33 } [get_ports { pwrScl }];
set_property -dict { PACKAGE_PIN J21 IOSTANDARD LVCMOS33 } [get_ports { pwrSda }];
set_property -dict { PACKAGE_PIN J22 IOSTANDARD LVCMOS33 } [get_ports { configScl }];
set_property -dict { PACKAGE_PIN J20 IOSTANDARD LVCMOS33 } [get_ports { configSda }];
set_property -dict { PACKAGE_PIN J19 IOSTANDARD LVCMOS33 } [get_ports { fdSerSdio }];
set_property -dict { PACKAGE_PIN K16 IOSTANDARD LVCMOS33 } [get_ports { tempAlertL }];

##############################################################################

# Timing Constraints 
create_clock -name pgpClkP -period 3.200 [get_ports { gtClkP[0] }]
create_clock -name evrClkP -period 2.691 [get_ports { gtClkP[1] }]

create_clock -name cbl0Half1Clk0 -period 10 [get_ports {cbl0Half1P[0]}]
create_clock -name cbl1Half0Clk0 -period 10 [get_ports {cbl1Half0P[0]}]
create_clock -name cbl1Half1Clk0 -period 10 [get_ports {cbl1Half1P[0]}]

create_generated_clock -name dnaClk     [get_pins {U_Core/U_FpgaSystem/U_AxiVersion/GEN_DEVICE_DNA.DeviceDna_1/GEN_7SERIES.DeviceDna7Series_Inst/BUFR_Inst/O}] 
create_generated_clock -name dnaClkInv  [get_pins {U_Core/U_FpgaSystem/U_AxiVersion/GEN_DEVICE_DNA.DeviceDna_1/GEN_7SERIES.DeviceDna7Series_Inst/DNA_CLK_INV_BUFR/O}] 
create_generated_clock -name iprogClk   [get_pins {U_Core/U_FpgaSystem/U_AxiVersion/GEN_ICAP.Iprog_1/GEN_7SERIES.Iprog7Series_Inst/DIVCLK_GEN.BUFR_ICPAPE2/O}] 

create_generated_clock -name refClk200 [get_pins -hier -filter {NAME =~ *.U_PGP/U_MMCM/MmcmGen.U_Mmcm/CLKOUT0}] 
create_generated_clock -name axilClk   [get_pins -hier -filter {NAME =~ *.U_PGP/U_MMCM/MmcmGen.U_Mmcm/CLKOUT1}] 

create_generated_clock -name cbl0Half1Clk1 [get_pins {U_Core/U_CLinkWrapper/U_ClinkTop/U_Cbl0Half1/U_DataShift/U_ClkGen/U_Mmcm/CLKOUT0}] 
create_generated_clock -name cbl0Half1Clk2 [get_pins {U_Core/U_CLinkWrapper/U_ClinkTop/U_Cbl0Half1/U_DataShift/U_ClkGen/U_Mmcm/CLKOUT1}] 

create_generated_clock -name cbl1Half0Clk1 [get_pins {U_Core/U_CLinkWrapper/U_ClinkTop/U_DualCtrlDis.U_Cbl1Half0/U_DataShift/U_ClkGen/U_Mmcm/CLKOUT0}] 
create_generated_clock -name cbl1Half0Clk2 [get_pins {U_Core/U_CLinkWrapper/U_ClinkTop/U_DualCtrlDis.U_Cbl1Half0/U_DataShift/U_ClkGen/U_Mmcm/CLKOUT1}] 

create_generated_clock -name cbl1Half1Clk1 [get_pins {U_Core/U_CLinkWrapper/U_ClinkTop/U_Cbl1Half1/U_DataShift/U_ClkGen/U_Mmcm/CLKOUT0}] 
create_generated_clock -name cbl1Half1Clk2 [get_pins {U_Core/U_CLinkWrapper/U_ClinkTop/U_Cbl1Half1/U_DataShift/U_ClkGen/U_Mmcm/CLKOUT1}] 

set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {dnaClk}] -group [get_clocks {dnaClkInv}] -group [get_clocks {iprogClk}] 

set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl0Half1Clk0}] 
set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl0Half1Clk1}] 
set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl0Half1Clk2}] 
set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl1Half0Clk0}] 
set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl1Half0Clk1}] 
set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl1Half0Clk2}] 
set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl1Half1Clk0}] 
set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl1Half1Clk1}] 
set_clock_groups -asynchronous -group [get_clocks {axilClk}] -group [get_clocks {refClk200}] -group [get_clocks {cbl1Half1Clk2}] 

# Clink input clock async to derived clocks (for clock input shift)
set_clock_groups -asynchronous -group [get_clocks {cbl0Half1Clk0}] -group [get_clocks {cbl0Half1Clk2}]
set_clock_groups -asynchronous -group [get_clocks {cbl1Half0Clk0}] -group [get_clocks {cbl1Half0Clk2}]
set_clock_groups -asynchronous -group [get_clocks {cbl1Half1Clk0}] -group [get_clocks {cbl1Half1Clk2}]

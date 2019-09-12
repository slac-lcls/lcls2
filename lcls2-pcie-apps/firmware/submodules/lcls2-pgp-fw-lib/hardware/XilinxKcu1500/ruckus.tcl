# Load RUCKUS environment and library
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

# Load local source Code 
loadSource      -dir  "$::DIR_PATH/rtl"
loadConstraints -path "$::DIR_PATH/xdc/Hardware.xdc"

# Case the timing on communication protocol
if { [info exists ::env(INCLUDE_PGP3_10G)] != 1 || $::env(INCLUDE_PGP3_10G) == 0 } {
   loadConstraints -path "$::DIR_PATH/xdc/Pgp2bTiming.xdc"
} else {
   loadConstraints -path "$::DIR_PATH/xdc/Pgp3Timing.xdc"
}

# Load shared source code
loadRuckusTcl "$::DIR_PATH/../../shared"

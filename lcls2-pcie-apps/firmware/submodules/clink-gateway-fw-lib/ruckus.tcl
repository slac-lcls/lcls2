# Load RUCKUS environment and library
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

# Check for Vivado version 2018.2 (or later)
if { [VersionCheck 2018.2 ] < 0 } {
   exit -1
}

# Check for submodule tagging
if { [info exists ::env(OVERRIDE_SUBMODULE_LOCKS)] != 1 || $::env(OVERRIDE_SUBMODULE_LOCKS) == 0 } {
   if { [SubmoduleCheck {ruckus}           {1.7.5}  ] < 0 } {exit -1}
   if { [SubmoduleCheck {surf}             {1.9.8}  ] < 0 } {exit -1}
} else {
   puts "\n\n*********************************************************"
   puts "OVERRIDE_SUBMODULE_LOCKS != 0"
   puts "Ignoring the submodule locks in clink-gateway-fw-lib/ruckus.tcl"
   puts "*********************************************************\n\n"
}

# Load local source Code 
loadSource      -dir "$::DIR_PATH/rtl"

# Load local source Code
loadConstraints -path "$::DIR_PATH/xdc/ClinkGateway.xdc"

# Case the timing on communication protocol
if { [info exists ::env(INCLUDE_PGP3_10G)] != 1 || $::env(INCLUDE_PGP3_10G) == 0 } {
   loadConstraints -path "$::DIR_PATH/xdc/Pgp2bTiming.xdc"
} else {
   loadConstraints -path "$::DIR_PATH/xdc/Pgp3Timing.xdc"
   
}

# Updating the impl_1 strategy
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

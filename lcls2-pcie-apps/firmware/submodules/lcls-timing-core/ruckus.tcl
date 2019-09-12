# Load RUCKUS environment and library
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

# Check for Vivado version 2016.4 (or later)
if { [VersionCheck 2016.4 ] < 0 } {
   exit -1
}

# Check for submodule tagging
if { [info exists ::env(OVERRIDE_SUBMODULE_LOCKS)] != 1 || $::env(OVERRIDE_SUBMODULE_LOCKS) == 0 } {
   if { [SubmoduleCheck {ruckus}           {1.7.5} ] < 0 } {exit -1}
   if { [SubmoduleCheck {surf}             {1.9.7} ] < 0 } {exit -1}
} else {
   puts "\n\n*********************************************************"
   puts "OVERRIDE_SUBMODULE_LOCKS != 0"
   puts "Ignoring the submodule locks in amc-carrier-core/ruckus.tcl"
   puts "*********************************************************\n\n"
}

# Load ruckus files
loadRuckusTcl "$::DIR_PATH/LCLS-I"  "quiet"
loadRuckusTcl "$::DIR_PATH/LCLS-II" "quiet"

# Check for submodule tagging
puts "*********************************************************"
if { [info exists ::env(TIMING_EXT_PKG)] != 1 || $::env(TIMING_EXT_PKG) == 0 } {
   puts "TIMING_EXT_PKG = 0"
   puts "Using default TimingExtnPkg.vhd in lcls-timing-core"
   loadSource -path "$::DIR_PATH/LCLS-II/core/rtl/default/TimingExtnPkg.vhd"
} else {
   puts "TIMING_EXT_PKG = 1"
   puts "Using application's defined TimingExtnPkg.vhd"
}
puts "*********************************************************"

# Load RUCKUS environment and library
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

# Check for Vivado version 2018.2 (or later)
if { [VersionCheck 2018.2 ] < 0 } {
   exit -1
}

# Check for submodule tagging
if { [info exists ::env(OVERRIDE_SUBMODULE_LOCKS)] != 1 || $::env(OVERRIDE_SUBMODULE_LOCKS) == 0 } {
   if { [SubmoduleCheck {axi-pcie-core}    {2.1.2}  ] < 0 } {exit -1}
   if { [SubmoduleCheck {lcls-timing-core} {1.12.6} ] < 0 } {exit -1}
   if { [SubmoduleCheck {ruckus}           {1.7.5}  ] < 0 } {exit -1}
   if { [SubmoduleCheck {surf}             {1.9.7}  ] < 0 } {exit -1}
} else {
   puts "\n\n*********************************************************"
   puts "OVERRIDE_SUBMODULE_LOCKS != 0"
   puts "Ignoring the submodule locks in lcls2-pgp-fw-lib/shared/ruckus.tcl"
   puts "*********************************************************\n\n"
}

# Load local source Code 
loadSource -dir "$::DIR_PATH/rtl"

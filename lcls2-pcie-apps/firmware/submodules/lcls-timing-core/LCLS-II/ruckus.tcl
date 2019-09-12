# Load RUCKUS library
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

# Load Source Code
loadRuckusTcl "$::DIR_PATH/core"
loadRuckusTcl "$::DIR_PATH/evr"

# Get the family type
set family [getFpgaFamily]

if { ${family} == "kintexu" } {
   loadRuckusTcl "$::DIR_PATH/gthUltraScale"
}

if { ${family} == "kintexuplus" } {
   loadRuckusTcl "$::DIR_PATH/gtyUltraScale+"
}

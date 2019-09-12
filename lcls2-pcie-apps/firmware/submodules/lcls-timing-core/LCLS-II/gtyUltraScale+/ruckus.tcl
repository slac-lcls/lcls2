# Load RUCKUS environment and library
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

if { $::env(VIVADO_VERSION) >= 2018.2 } {

   loadSource -dir "$::DIR_PATH/rtl"

   loadSource   -path "$::DIR_PATH/coregen/TimingGty_extref.dcp"
   # loadIpCore -path "$::DIR_PATH/coregen/TimingGty_extref.xci"

   loadSource   -path "$::DIR_PATH/coregen/TimingGty_fixedlat.dcp"
   # loadIpCore -path "$::DIR_PATH/coregen/TimingGty_fixedlat.xci"

} else {
   puts "\n\nWARNING: $::DIR_PATH requires Vivado 2018.2 (or later)\n\n"
}
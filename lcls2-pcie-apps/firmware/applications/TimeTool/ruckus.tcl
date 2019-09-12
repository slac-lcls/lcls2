# Load RUCKUS environment and library
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

# Load local Source Code and Constraints
loadSource      -dir "$::DIR_PATH/hdl"
# loadIpCore -path "$::DIR_PATH/coregen/ila_1.xci"
loadIpCore -path "$::DIR_PATH/coregen/fir_compiler_3.xci"
loadIpCore -path "$::DIR_PATH/coregen/fir_compiler_0.xci"
loadIpCore -path "$::DIR_PATH/coregen/fir_compiler_1.xci"    #the one to be used in deployment. has reloadable coefficients

# Load Simulation
loadSource -sim_only -dir "$::DIR_PATH/tb"


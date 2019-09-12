# Load RUCKUS environment and library
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

# Load base sub-modules
loadRuckusTcl $::env(PROJ_DIR)/../../submodules/surf
loadRuckusTcl $::env(PROJ_DIR)/../../submodules/lcls-timing-core
loadRuckusTcl $::env(PROJ_DIR)/../../submodules/axi-pcie-core/hardware/XilinxKcu1500
loadRuckusTcl $::env(PROJ_DIR)/../../submodules/lcls2-pgp-fw-lib/hardware/XilinxKcu1500
loadRuckusTcl $::env(PROJ_DIR)/../../applications

# Load the l2si-core source code
loadSource -dir "$::env(PROJ_DIR)/../../submodules/l2si-core/xpm/rtl"
loadSource -dir "$::env(PROJ_DIR)/../../submodules/l2si-core/base/rtl"

# Load local source Code and constraints
loadSource -dir "$::DIR_PATH/../../targets/TimeToolKcu1500/hdl"
set_property top {TimeToolKcu1500} [get_filesets {sources_1}]

# Load Simulation
loadSource -sim_only -dir "$::DIR_PATH/tb"

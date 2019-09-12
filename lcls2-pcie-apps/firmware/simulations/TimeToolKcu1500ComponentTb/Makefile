# List all the components level simulation testbeds to run
LIST_COMP_TB=AxiVersionTb AxiLiteAsyncTb FifoFwftTb

# Set the simulation duration
export VIVADO_PROJECT_SIM_TIME = 100 us

# Define Firmware Version Number
export PRJ_VERSION = 0x00000001

# Define target part
export PRJ_PART = xcku115-flvb2104-2-e

#simulation package export
export TIMING_EXT_PKG = 1

# By creating a separate target for each submake, the top level 'make -j' can build them in parallel!
.PHONY : $(LIST_COMP_TB)
all: $(LIST_COMP_TB)
$(LIST_COMP_TB):
	@$(MAKE) PROJECT=$@ clean
	@$(MAKE) PROJECT=$@ VIVADO_PROJECT_SIM=$@ xsim

# Use top level makefile
include ../../submodules/ruckus/system_vivado.mk

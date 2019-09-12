export GIT_BYPASS = 1
##########################################################################################
# Applications must define the 32-bitFirmware Version Number
# PRJ_VERSION = 0x00000013 - Adopt TimingExtnPkg and factorization
# PRJ_VERSION = 0x00000014 - Automate front panel link rx resets, fix paddr latching
# PRJ_VERSION = 0x00000015 - Add readout group registers and LCLS1 timing input
# PRJ_VERSION = 0x00000016 - Add cuRxEnable/usRxEnable
# PRJ_VERSION = 0x00000017 - Migrate to l2si-core.  Add link rxTimeOut.  Add link fulltime counts
# PRJ_VERSION = 0x00000018 - Add link groupMask.  Remove linkTxDelay, linkPartition.  
#                          - Change tmit PADDR to be configured via register access.
# PRJ_VERSION = 0x00000019 - Use obBsaMaster stream for sequence checkpoint notification
# PRJ_VERSION = 0x0000001A - Add parsing of partition messages for group link clear
##########################################################################################
export PRJ_VERSION = 0x00000000

##########################################################################################
# Applications must define if using advance AMC carrier build
# 
# AMC_ADV_BUILD = 0: Uses pre-built AmcCarrierCore.vhd image (99% case)
# AMC_ADV_BUILD = 1: Allows full configuration of AmcCarrierCore generics
#
# Here is a possible list of reason why you would need AMC_ADV_BUILD = 1
#     1) Timing Generator application
#     2) Not using BSA engine and would like to free up resources
#     3) Using zone3 (RTM) instead of zone2 (backplane) for Ethernet
#
# Note: (AMC_ADV_BUILD = 1) requires a munch longer build time
##########################################################################################

# COMMON_FILE is defined by application
export COMMON_FILE = xpm

##########################################################################################
# Application have the option to remove unused source code from Vivado project
#     If REMOVE_UNUSED_CODE = 0, all source code is kept after importing to Vivado project
#     If REMOVE_UNUSED_CODE = 1, all unused source code is removed from Vivado project
##########################################################################################
export REMOVE_UNUSED_CODE = 1

##########################################################################################
############################  DO NOT EDIT THE CODE BELOW  ################################
##########################################################################################

# Define target output
target: prom

# Define target part
export PRJ_PART = XCKU040-FFVA1156-2-E
export BOARD_PART = 
export PRJ_FMC =

# Use top level makefile
include ../../submodules/ruckus/system_vivado.mk

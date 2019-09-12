##############################################################################
## This file is part of 'ATLAS RD53 DEV'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'ATLAS RD53 DEV', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

##############################
# Get variables and procedures
##############################
source -quiet $::env(RUCKUS_DIR)/vivado_env_var.tcl
source -quiet $::env(RUCKUS_DIR)/vivado_proc.tcl

# Bypass the debug chipscope generation
return

############################
## Open the synthesis design
############################
open_run synth_1

###############################
## Set the name of the ILA core
###############################
set ilaName u_ila_1

##################
## Create the core
##################
CreateDebugCore ${ilaName}

#######################
## Set the record depth
#######################
set_property C_DATA_DEPTH 1024 [get_debug_cores ${ilaName}]

#################################
## Set the clock for the ILA core
#################################
SetDebugCoreClk ${ilaName} {U_ClinkTop/U_Cbl0Half1/clinkClk}

#######################
## Set the debug Probes
#######################

ConfigProbe ${ilaName} {U_ClinkTop/U_Cbl0Half1/}
#ConfigProbe ${ilaName} {U_TimeToolCore/r[axilReadSlave][*}
#ConfigProbe ${ilaName} {U_TimeToolCore/axilWriteMaster[*}
#ConfigProbe ${ilaName} {U_TimeToolCore/r[axilWriteSlave][*}

# ConfigProbe ${ilaName} {U_TimeToolCore/r[addValue][*]}
# ConfigProbe ${ilaName} {U_TimeToolCore/r[dialInOpCode][*]}
# ConfigProbe ${ilaName} {U_TimeToolCore/r[dialInTriggerDelay][*]}

##########################
## Write the port map file
##########################
WriteDebugProbes ${ilaName} 

